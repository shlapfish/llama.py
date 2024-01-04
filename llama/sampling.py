import math
from copy import copy
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from bindings import lib, ffi
from llama.context import Context, Logits, Sequence
from llama.model import Model

TokenData = np.dtype([('token', 'i4'), ('logit', 'f4'), ('p', 'f4')])


@dataclass(slots=True, frozen=True)
class TokenProbability:
    token: int
    log_odds: float
    prob: float
    logit: float  # not log-odds, but the output of the LLM before softmax


@dataclass(slots=True)
class TokenDataArray:
    token_data: np.ndarray[TokenData]
    ltdas: Any
    """llama_token_data_array struct"""

    def __init__(self, model: Model):
        self.token_data = np.zeros(model.vocab_size, dtype=TokenData)
        self.ltdas = ffi.new("struct llama_token_data_array *")
        self.ltdas.data = ffi.from_buffer("llama_token_data *", self.token_data.data)

    def insert(self, logits: Logits):
        logits_arr = logits.get()
        assert self.token_data.shape[0] <= logits_arr.shape[0]
        self.ltdas.sorted = False
        self.ltdas.size = logits_arr.shape[0]
        self.token_data['token'] = np.arange(self.token_data.shape[0], dtype=np.int32)
        self.token_data['logit'] = logits_arr

    def probabilities(self, context: Context, top_n: int = None) -> list[TokenProbability]:
        lib.llama_sample_softmax(context._raw, self.ltdas)

        probs = []
        for i in range(top_n or self.ltdas.size):
            tmp = self.ltdas.data[i]
            probs.append(TokenProbability(
                token=tmp.id,
                log_odds=math.log(tmp.p),
                logit=tmp.logit,
                prob=tmp.p,
            ))
        return probs

    def __copy__(self):
        new = TokenDataArray.__new__(type(self))
        new.token_data = self.token_data.copy()
        new.ltdas = ffi.new("struct llama_token_data_array *")
        ffi.memmove(new.ltdas, self.ltdas, ffi.sizeof("struct llama_token_data_array"))
        new.ltdas.data = ffi.from_buffer("llama_token_data *", new.token_data.data)
        return new


@dataclass
class Mirostatv2Sampler:
    context: Context
    target_entropy: float = 5.
    update_speed: float = 0.1

    def __post_init__(self):
        self.mu = ffi.new("float *", 2 * self.target_entropy)
        self.tda = TokenDataArray(self.context.model)

    def sample(self, logits: Logits) -> int:
        self.tda.insert(logits)
        return lib.llama_sample_token_mirostat_v2(
            self.context._raw, self.tda.ltdas, self.target_entropy, self.update_speed, self.mu
        )

    def __copy__(self):
        ret = Mirostatv2Sampler.__new__(type(self))
        ret.context = self.context
        ret.target_entropy = self.target_entropy
        ret.update_speed = self.update_speed
        ret.mu = ffi.new("float *", self.mu[0])
        ret.tda = copy(self.tda)
        return ret


@dataclass(slots=True)
class Node:
    logit_sum: float
    log_odds: float
    token: int
    seq: Sequence
    children: tuple['Node', ...] = ()

    def count_nodes(self) -> int:
        return 1 + sum(c.count_nodes() for c in self.children)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(slots=True)
class TreeSampler:
    context: Context
    model: Model
    tree_size: int
    deepening_factor: float = 0.8  # ]0, 1]  higher mean looking further ahead, but less breadth
    branching_factor: int = 20  # maybe a probability cutoff?
    max_leaves: int = 8000
    tda: TokenDataArray = None

    def __post_init__(self):
        self.tda = TokenDataArray(self.model)

    def compute_children(self, nodes: list['Node']) -> Iterator['Node']:
        new_seqs = [n.seq.duplicate() for n in nodes]
        new_logits = [s.insert(n.token) for s, n in zip(new_seqs, nodes)]
        self.context.process()
        for seq, logits, node in zip(new_seqs, new_logits, nodes):
            self.tda.insert(logits)
            node.children = tuple(
                Node(
                    logit_sum=node.logit_sum + prob.logit,
                    log_odds=node.log_odds + prob.log_odds,
                    token=prob.token,
                    seq=seq,
                ) for prob in self.tda.probabilities(self.context, top_n=self.branching_factor)
            )
            yield from node.children

    def node_score(self, node: 'Node') -> float:
        return math.pow(1/self.deepening_factor, node.log_odds) * node.logit_sum

    def grow_leaf_list(self, leaf_list: list['Node']) -> list['Node']:
        """
        :returns: The nodes that are no longer leaves.
        """
        # The KV cache gets fragmented.
        view = lib.llama_kv_cache_view_init(self.context._raw, len(Sequence.used_ids))
        lib.llama_kv_cache_view_update(self.context._raw, ffi.addressof(view))
        contiguous_batch_size = view.max_contiguous
        # contiguous_batch_size = 64
        ## fields = ("n_cells", "n_max_seq", "token_count", "used_cells", "max_contiguous")
        ## print("\n".join(f"{field}: {getattr(view, field)}" for field in fields))
        lib.llama_kv_cache_view_free(ffi.addressof(view))

        batch = min(self.context.max_batch_size, contiguous_batch_size)
        best_nodes = [leaf_list.pop() for _ in range(min(batch, len(leaf_list)))]

        new_leafs = self.compute_children(best_nodes)
        leaf_list.extend(new_leafs)
        leaf_list.sort(key=lambda n: self.node_score(n))

        return best_nodes

    def print_node(self, node: Node, ignore_prefix: str, recurse=False):
        text = str(node.seq).removeprefix(ignore_prefix) + self.model.detokenize(node.token)
        print(f"{node.count_nodes():5} {self.node_score(node):8.3f} {text}")
        if recurse:
            for child in node.children:
                self.print_node(child, ignore_prefix)

    def generate(self, seq: Sequence, logits: Logits) -> Iterator[str]:
        tda = TokenDataArray(self.model)
        tda.insert(logits)
        leaf_list = [
            Node(
                logit_sum=prob.logit,
                log_odds=prob.log_odds,
                token=prob.token,
                seq=seq,
            ) for prob in tda.probabilities(self.context, top_n=self.branching_factor)
        ]

        roots = list(leaf_list)
        non_leaves = 0

        while True:
            while non_leaves < self.tree_size:
                processed_nodes = self.grow_leaf_list(leaf_list)
                non_leaves += len(processed_nodes)
                leaf_list = leaf_list[-self.max_leaves:]
            #     for node in processed_nodes:
            #         self.print_node(node, prompt)
            #
            # for node in reversed(roots):
            #     self.print_node(node, prompt)

            # trim the tree again by picking a root node.
            # we pick the one with the most nodes (average length might be better)
            pick = max(roots, key=Node.count_nodes)
            # self.print_node(pick, prompt)
            roots.remove(pick)
            to_remove = set(roots)
            for node in roots:
                node.seq.clear()
                to_remove.add(node)
                if node.children:
                    roots.extend(node.children)
                    non_leaves -= 1

            yield self.model.detokenize(pick.token)

            leaf_list = list(filter(lambda n: n not in to_remove, leaf_list))
            roots = list(pick.children)
