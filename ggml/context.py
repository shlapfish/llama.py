from dataclasses import dataclass
from functools import cached_property

from bindings import ffi, lib, StructWrapper, c_str
import ggml


_ctx_stack = []


class Context(StructWrapper):

    @cached_property
    def tensors(self) -> 'TensorsView':
        return TensorsView(self)

    @staticmethod
    def new(mem_size: int):
        params = ffi.new("struct ggml_init_params *")
        params.mem_size = mem_size
        params.mem_buffer = ffi.NULL
        params.no_alloc = False
        return Context(lib.ggml_init(params[0]))

    @staticmethod
    def current() -> 'Context':
        if not _ctx_stack:
            raise Exception("Not in a context.")
        return _ctx_stack[-1]

    def free(self):
        lib.ggml_free(self)
        self.struct = None

    def __enter__(self):
        _ctx_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _ctx_stack.pop() is not self:
            raise Exception("Bad context nesting. What are you even trying to do???")
        self.free()


@dataclass(slots=True)
class TensorsView:
    ctx: Context

    def __getitem__(self, name) -> 'ggml.Tensor':
        ptr = lib.ggml_get_tensor(self.ctx, c_str(name))
        if ptr:
            return ggml.Tensor(ptr, ctx=self.ctx)
        else:
            raise Exception(f"No tensor with name {name} in context.")

    def __iter__(self):
        tensor = lib.ggml_get_first_tensor(self.ctx)
        while tensor:
            yield ggml.Tensor(tensor, ctx=self.ctx)
            tensor = lib.ggml_get_next_tensor(self.ctx, tensor)
