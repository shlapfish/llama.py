from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np
from src.bindings import lib, ffi
from src.model import Model
from src.context import Context, Logits

TokenData = np.dtype([('token', 'i4'), ('logit', 'f4'), ('p', 'f4')])

# The sampling interface is going to change, so for now I only implemented Mirostatv2


@dataclass
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

    def softmax(self, context: Context):
        lib.llama_sample_softmax(context._raw, self.ltdas)

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


class Node:
    tokens: dict[int, 'Node']

    def __init__(self):
        self.tokens = defaultdict(Node)

    def insert(self, tokens: list[int]):
        first, *rest = tokens
        self.tokens[first].insert(rest)

    @staticmethod
    def build_tree(model: Model, possibilities: list[str]) -> 'Node':
        node = Node()
        for option in possibilities:
            node.insert(model.tokenize(option))
        return node


@dataclass
class SetSampler:
    """Only allow a set of possible answers."""
    node: Node
    tda: TokenDataArray

    def __init__(self, tree: Node, model: Model):
        self.node = tree
        self.tda = TokenDataArray(model)

    def sample(self, logits: Logits) -> int:
        self.tda.ltdas.sorted = False
        self.tda.ltdas.size = len(self.node.tokens)
        tokens = np.array(self.node.tokens.keys(), dtype=np.int32)
        self.tda.token_data['token'] = tokens
        self.tda.token_data['logit'] = logits.get()[tokens]

    def accept(self, token: int):
        assert token in self.node.tokens
        self.node = self.node.tokens[token]

    def end(self) -> bool:
        return not self.node.tokens