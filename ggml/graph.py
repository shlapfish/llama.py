from dataclasses import dataclass

from bindings import lib, StructWrapper
from ggml import Context, Tensor


@dataclass
class Graph(StructWrapper):
    ctx: Context

    def __init__(self, size=lib.GGML_DEFAULT_GRAPH_SIZE, grads=False):
        self.struct = lib.ggml_new_graph_custom(Context.current(), size, grads)
        self.ctx = Context.current()

    def build_forward_expand(self, t: Tensor):
        lib.ggml_build_forward_expand(self, t)

    def compute(self, threads: int = lib.GGML_DEFAULT_N_THREADS):
        lib.ggml_graph_compute_with_ctx(self.ctx, self, threads)

    def print(self):
        lib.ggml_graph_print(self)

    def __len__(self):
        return self.struct.size
