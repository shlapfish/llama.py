from enum import IntEnum

from llama import lib, ffi, load_libllama
from src.bindings import c_str

load_libllama("../llama.cpp")

lib.ggml_time_init()


class Type(IntEnum):
    F32 = lib.GGML_TYPE_F32
    F16 = lib.GGML_TYPE_F16
    I8 = lib.GGML_TYPE_I8
    I16 = lib.GGML_TYPE_I16
    I32 = lib.GGML_TYPE_I32

    def is_float(self) -> bool:
        return self == Type.F32

    def is_int(self) -> bool:
        return self in (Type.I8, Type.I16, Type.I32)


class Context:
    def __init__(self, mem_size: int):
        params = ffi.new("struct ggml_init_params *")
        params.mem_size = mem_size
        params.mem_buffer = ffi.NULL
        self._raw = lib.ggml_init(params[0])

    def new_tensor(self, size: list[int], type: Type, name: str = None, is_param: bool = False):
        size_arr = ffi.new("int64_t []", size)
        ptr = lib.ggml_new_tensor(self._raw, type, len(size), size_arr)
        ret = Tensor(self, ptr)
        if name:
            ret.name = name
        if is_param:
            ret.set_param()
        return ret

    def new_graph(self):
        return Graph(self, lib.ggml_new_graph(self._raw))

    def free(self):
        lib.ggml_free(self._raw)
        self._raw = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()


class Tensor:
    def __init__(self, ctx: Context, raw_ptr):
        self.ctx = ctx
        self._raw = raw_ptr

    @property
    def name(self):
        return ffi.string(lib.ggml_get_name(self._raw)).decode()

    @name.setter
    def name(self, name: str):
        lib.ggml_set_name(self._raw, c_str(name))

    @property
    def type(self) -> Type:
        return Type(self._raw.type)

    def set_param(self):
        lib.ggml_set_param(self.ctx._raw, self._raw)

    def __mul__(self, other: 'Tensor'):
        return Tensor(self.ctx, lib.ggml_mul(self.ctx._raw, self._raw, other._raw))

    def __add__(self, other: 'Tensor'):
        return Tensor(self.ctx, lib.ggml_add(self.ctx._raw, self._raw, other._raw))

    def set(self, val) -> 'Tensor':
        match val:
            case float(f): raw = lib.ggml_set_f32(self._raw, f)
            case int(i): raw = lib.ggml_set_i32(self._raw, i)
            case _: raise Exception("Dunno what to do with that type yet. Pls implement.")
        return Tensor(self.ctx, raw)

    def __getitem__(self, item: int | tuple[int, ...]):
        # TODO check the order of the indices, I might've gotten them wrong
        if not isinstance(item, tuple):
            item = (item,)
        # insert 0s up to 4 dimensions because the ggml_get functions are hardcoded to 4 dimensions
        item += (4 - len(item)) * (0,)
        if self.type.is_float():
            return lib.ggml_get_f32_nd(self._raw, *item)
        elif self.type.is_int():
            return lib.ggml_get_i32_nd(self._raw, *item)
        else:
            raise Exception(f"Cannot convert type {self.type.name} to a python-readable format.")


class Graph:
    def __init__(self, ctx: Context, raw_ptr):
        self.ctx = ctx
        self._raw = raw_ptr

    def build_forward_expand(self, t: Tensor):
        lib.ggml_build_forward_expand(self._raw, t._raw)

    def compute(self, threads: int = lib.GGML_DEFAULT_N_THREADS):
        lib.ggml_graph_compute_with_ctx(self.ctx._raw, self._raw, threads)

    def print(self):
        lib.ggml_graph_print(self._raw)


MiB = 1024*1024
with Context(16*MiB) as ctx:
    x = ctx.new_tensor([1], Type.F32, "x", is_param=True)
    a = ctx.new_tensor([1], Type.F32, "a")
    b = ctx.new_tensor([1], Type.F32, "b")

    f = a * x*x + b

    graph = ctx.new_graph()
    graph.build_forward_expand(f)

    # set the input variable and parameter values
    x.set(2.)
    a.set(3.)
    b.set(4.)

    graph.compute()

    result = f[0]

    graph.print()

print(result)
