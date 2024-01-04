from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property

from bindings import lib, StructWrapper, ffi, c_str
from ggml import Context
from ggml.context import TensorsView


class GgufDType(IntEnum):
    bool = lib.GGUF_TYPE_BOOL

    u8 = lib.GGUF_TYPE_UINT8
    u16 = lib.GGUF_TYPE_UINT16
    u32 = lib.GGUF_TYPE_UINT32
    u64 = lib.GGUF_TYPE_UINT64

    i8 = lib.GGUF_TYPE_INT8
    i16 = lib.GGUF_TYPE_INT16
    i32 = lib.GGUF_TYPE_INT32
    i64 = lib.GGUF_TYPE_INT64

    f32 = lib.GGUF_TYPE_FLOAT32
    f64 = lib.GGUF_TYPE_FLOAT64

    str = lib.GGUF_TYPE_STRING
    arr = lib.GGUF_TYPE_ARRAY

    def is_scalar(self):
        return self.name in ("u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "u64", "i64", "f64")


class InfoView(StructWrapper):
    def __getitem__(self, item: str):
        key = lib.gguf_find_key(self, c_str(item))
        if key == -1:
            raise KeyError()
        gguf_type = GgufDType(lib.gguf_get_kv_type(self, key))
        if gguf_type == lib.GGUF_TYPE_STRING:
            return ffi.string(lib.gguf_get_val_str(self, key)).decode()
        elif gguf_type.is_scalar():
            return getattr(lib, f"gguf_get_val_{gguf_type.name}")(self, key)
        else:
            raise Exception(f"Info type '{gguf_type.name}' not yet implemented.")

    def __iter__(self):
        for key in range(lib.gguf_get_n_kv(self)):
            yield ffi.string(lib.gguf_get_key(self, key)).decode()


@dataclass
class Gguf(StructWrapper):
    ctx: Context

    @classmethod
    def load_from_file(cls, path: str) -> 'Gguf':
        params = ffi.new("struct gguf_init_params *")
        params.no_alloc = False
        params.ctx = ffi.new("struct ggml_context **", ffi.cast("void *", 1))
        struct = lib.gguf_init_from_file(c_str(path), params[0])
        return cls(struct, Context(params.ctx[0]))

    @property
    def tensors(self) -> TensorsView:
        return self.ctx.tensors

    @cached_property
    def info(self) -> InfoView:
        return InfoView(self.struct)

    def get_names(self) -> list[str]:
        return [ffi.string(lib.gguf_get_tensor_name(self, i)).decode()
                for i in range(lib.gguf_get_n_tensors(self))]
