from copy import copy
from dataclasses import dataclass, field
from itertools import count
from typing import Any, ClassVar, Iterable

import numpy as np

from src.bindings import lib, ffi, copy_array, UseAfterFree
from src.model import Model


@dataclass(slots=True)
class Logits:
    _raw: Any

    def get(self) -> np.ndarray[np.float32]:
        self._check()
        return np.frombuffer(ffi.buffer(self._raw), dtype=np.float32)

    def _check(self):
        if self._raw is None:
            raise Exception("You have to run context.process() first to fill in the logits.")


@dataclass(slots=True, frozen=True)
class Insert:
    seq_id: int
    pos: int
    token: int
    logits: Logits | None


class Context:
    """
    - seed: `int` --- RNG seed, -1 for random.
    - context_size: `int` --- Size of the context window in tokens. Default = take from model.
    - batch_size: `int` --- Prompt processing maximum batch size.
    - threads: `int` --- Number of threads to used.
    - embedding: `bool` --- Embedding mode only.
    """
    def __init__(self, model: Model, context_size=None, threads=None, batch_size=None, **kwargs):
        self.planned_inserts: list[Insert] = []
        self.model = model  # we need to keep this alive
        params = lib.llama_context_default_params()

        params.n_ctx = context_size if context_size else 0
        if batch_size:
            params.n_batch = batch_size
        for k, v in kwargs.items():
            if k not in dir(params):
                raise ValueError(f"There is no parameter {k}.")
            setattr(params, k, v)

        self.max_batch_size = params.n_batch
        self._raw = lib.llama_new_context_with_model(model._raw, params)
        if self._raw == ffi.NULL:
            raise Exception("Could not create new context.")
        if threads:
            lib.llama_set_n_threads(self._raw, threads, threads)

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()

    def free(self):
        """Unloads the context, making it unusable."""
        lib.llama_free(self._raw)
        self._raw = None

    def __del__(self):
        try:
            if self._raw is not None:
                self.free()
        except UseAfterFree:
            pass


@dataclass
class Sequence:
    context: Context
    seq_id: int = field(hash=True)
    tokens: list[int]
    used_ids: ClassVar = set()
    original_length: int = 0  # so that clear() doesn't remove the parent

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
        assert amount <= l - self.original_length, f"Can truncate at most {l - self.original_length}."
        lib.llama_kv_cache_seq_rm(self.context._raw, self.seq_id, l - amount, -1)
        self.tokens = self.tokens[:l - amount]

    def duplicate(self) -> 'Sequence':
        """Duplicates a sequence and returns the new id. This doesn't use additional memory."""
        new_seq = Sequence(self.context)
        lib.llama_kv_cache_seq_cp(self.context._raw, self.seq_id, new_seq.seq_id, 0, len(self.tokens))
        new_seq.tokens = copy(self.tokens)
        new_seq.original_length = len(self.tokens)
        return new_seq

    def clear(self):
        """Removes tokens that were inserted. It does not remove tokens copied from the original sequence."""
        self.truncate_end(len(self.tokens) - self.original_length)
        self.tokens = self.tokens[:self.original_length]

    def insert(self, text_or_tokens: str | int | Iterable[int], generate_logits: bool = True) -> Logits | None:
        """
        Schedules token inserts and possibly returns logits for the last token. Initially the logits object will be empty.
        Run context.process() to execute all planned inserts, which will fill it in.
        """
        match text_or_tokens:
            case int(token):
                ret = Logits(None) if generate_logits else None
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
        if len(self.tokens) > self.original_length:
            print("You have to clear your sequences to clean up kv cache space.")
        self.used_ids.remove(self.seq_id)

    def __enter__(self):
        return self

    def __copy__(self):
        return self.duplicate()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()

    def __hash__(self):
        return self.seq_id

    def __str__(self):
        return self.context.model.detokenize(self.tokens)

    def __len__(self):
        return len(self.tokens)