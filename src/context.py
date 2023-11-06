from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from itertools import count
from typing import Any, ClassVar, Iterable

from src.bindings import lib, c_str, ffi, load_libllama, initialize_backend, copy_array
from src.model import Model


@dataclass(slots=True)
class Logits:
    _raw: Any
    _len: int

    def get(self) -> list[float]:
        self._check()
        return list(self._raw[0:self._len])

    def _check(self):
        if self._raw is None:
            raise Exception("You have to run context.process() first to fill in the logits.")

    def __len__(self):
        self._check()
        return self._len


@dataclass(slots=True, frozen=True)
class Insert:
    seq_id: int
    pos: int
    token: int
    logits: Logits | None


class Context:
    """
    - seed: `int` --- RNG seed, -1 for random.
    - n_ctx: `int` --- Size of the context window in tokens.
    - n_batch: `int` --- Prompt processing maximum batch size.
    - n_threads: `int` --- Number of threads to use for generation.
    - n_threads_batch: `int` --- Number of threads to use for batch processing.
    - embedding: `bool` --- Embedding mode only.
    """
    def __init__(self, model: Model, **kwargs):
        self.planned_inserts: list[Insert] = []
        self.model = model  # we need to keep this alive
        params = lib.llama_context_default_params()
        for k, v in kwargs.items():
            setattr(params, k, v)

        self.max_batch_size = params.n_batch
        self._raw = lib.llama_new_context_with_model(model._raw, params)
        if self._raw == ffi.NULL:
            raise Exception("Could not create new context.")

    def process(self):
        """Process all the pending insert jobs."""
        while self.planned_inserts:
            inserts = self.planned_inserts[:self.max_batch_size]
            n_seq_max = 1
            batch = lib.llama_batch_init(len(inserts), 0, n_seq_max)
            batch.n_tokens = len(inserts)
            n_vocab = self.model.vocab_size
            for i, insert in enumerate(inserts):
                batch.token[i] = insert.token
                batch.pos[i] = insert.pos
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = insert.seq_id
                batch.logits[i] = int(insert.logits is not None)

            result = lib.llama_decode(self._raw, batch)
            try:
                if result == 1: raise Exception("Could not find a KV slot for the batch (try reducing the size of the batch or increase the context)")
                if result > 1: raise Exception("Unknown error while processing a batch.")
                self.planned_inserts = self.planned_inserts[len(inserts):]
                for i, insert in enumerate(inserts):
                    if insert.logits is not None:
                        insert.logits._raw = copy_array(lib.llama_get_logits_ith(self._raw, i), n_vocab, "float")
            finally:
                lib.llama_batch_free(batch)

    def context_size(self) -> int:
        """The amount of tokens that can fit in the context window."""
        return lib.llama_n_ctx(self._raw)

    def __getattribute__(self, item):
        if item == "_raw" or self._raw is not None:
            return object.__getattribute__(self, item)
        else:
            raise Exception("Cannot use context after free.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()

    def free(self):
        """Unloads the context, making it unusable."""
        lib.llama_free(self._raw)
        self._raw = None

    def __del__(self):
        if self._raw is not None:
            self.free()


@dataclass
class Sequence:
    context: Context
    seq_id: int = field(hash=True)
    tokens: list[int]
    used_ids: ClassVar = set()

    def __init__(self, context: Context):
        self.context = context
        # first unused id
        self.seq_id = next(i for i in count() if i not in self.used_ids)
        self.used_ids.add(self.seq_id)
        self.tokens = []

    def truncate_begin(self, amount: int):
        """Removes the first `amount` tokens from the sequence."""
        assert amount <= len(self.tokens)
        lib.llama_kv_cache_seq_rm(self.context._raw, self.seq_id, 0, amount)
        lib.llama_kv_cache_seq_shift(self.context._raw, self.seq_id, amount, len(self.tokens), amount)
        self.tokens = self.tokens[amount:]

    def truncate_end(self, amount: int):
        """Remove the last `amount` tokens from the sequence."""
        l = len(self.tokens)
        assert amount <= l
        lib.llama_kv_cache_seq_rm(self.context._raw, self.seq_id, l - amount, -1)
        self.tokens = self.tokens[:l - amount]

    def duplicate(self) -> 'Sequence':
        """Duplicates a sequence and returns the new id. This doesn't use additional memory."""
        new_seq = Sequence(self.context)
        lib.llama_kv_cache_seq_cp(self.context._raw, self.seq_id, new_seq.seq_id, 0, len(self.tokens))
        new_seq.tokens = self.tokens
        return new_seq

    def clear(self):
        lib.llama_kv_cache_seq_rm(self.context._raw, self.seq_id, 0, -1)
        self.tokens.clear()

    def insert(self, text_or_tokens: str | int | Iterable[int], generate_logits: bool = False) -> Logits | None:
        """
        Schedules token inserts and possibly returns logits for the last token. Initially the logits object will be empty.
        Run context.process() to execute all planned inserts, which will fill it in.
        """
        match text_or_tokens:
            case int(token):
                ret = Logits(None, self.context.model.vocab_size) if generate_logits else None
                self.context.planned_inserts.append(Insert(
                    seq_id=self.seq_id,
                    pos=len(self.tokens),
                    token=token,
                    logits=ret
                ))
                self.tokens.append(token)
                return ret
            case str(text):
                return self.insert(self.context.model.tokenize(text), generate_logits)
            case [*tokens, int(last_token)]:
                for token in tokens:
                    self.insert(token)
                return self.insert(last_token, generate_logits)
            case []:
                assert not generate_logits
                return
            case _:
                raise ValueError("Unsupported type.")

    def __del__(self):
        self.used_ids.remove(self.seq_id)

    def __hash__(self):
        return self.seq_id

    def __str__(self):
        return self.context.model.detokenize(self.tokens)