from dataclasses import dataclass
from typing import Any

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