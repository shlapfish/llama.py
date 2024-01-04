from dataclasses import dataclass
from enum import IntEnum
from contextlib import contextmanager
from typing import Optional, Any
from math import prod

import numpy as np

from bindings import ffi, c_str, lib, StructWrapper
from ggml import Context


class DType(IntEnum):
    # floats
    f16 = lib.GGML_TYPE_F16
    f32 = lib.GGML_TYPE_F32

    # ints
    i8 = lib.GGML_TYPE_I8
    i16 = lib.GGML_TYPE_I16
    i32 = lib.GGML_TYPE_I32

    # k-quants
    q2_k = lib.GGML_TYPE_Q2_K
    q3_k = lib.GGML_TYPE_Q3_K
    q4_k = lib.GGML_TYPE_Q4_K
    q5_k = lib.GGML_TYPE_Q5_K
    q6_k = lib.GGML_TYPE_Q6_K
    q8_k = lib.GGML_TYPE_Q8_K

    # old quantized
    q4_0 = lib.GGML_TYPE_Q4_0
    q4_1 = lib.GGML_TYPE_Q4_1
    q5_0 = lib.GGML_TYPE_Q5_0
    q5_1 = lib.GGML_TYPE_Q5_1
    q8_0 = lib.GGML_TYPE_Q8_0
    q8_1 = lib.GGML_TYPE_Q8_1

    def as_numpy_type(self) -> np.dtype:
        match self:
            case DType.f32: return np.float32
            case DType.f16: return np.float16
            case DType.i32: return np.int32
            case DType.i16: return np.int16
            case DType.i8:  return np.int8
            case _:         raise Exception(f"Type {self} not supported by numpy.")

    @property
    def size(self) -> int:
        match self:
            case DType.f32 | DType.i32: return 4
            case DType.f16 | DType.i16: return 2
            case DType.i8: return 1
            case _: raise Exception(f"Type {self} does not have a defined item size.")

    def is_float(self) -> bool:
        return self == DType.f32

    def is_int(self) -> bool:
        return self in (DType.i8, DType.i16, DType.i32)

    def ffi_cast(self, void_ptr) -> Any:
        match self:
            case DType.f32: return ffi.cast("float *", void_ptr)
            case DType.i32: return ffi.cast("int *", void_ptr)
            case DType.i16: return ffi.cast("short *", void_ptr)
            case DType.i8:  return ffi.cast("char *", void_ptr)
            case _:         raise Exception(f"Cannot cast to type {self}.")

    def __str__(self):
        return self.name


name_prefixes = []


@contextmanager
def name_prefix(prefix: str):
    name_prefixes.append(prefix)
    yield
    name_prefixes.pop()


@dataclass
class Tensor(StructWrapper):
    ctx: Context

    def __init__(self, struct, name: str = None, ctx: Context = None):
        if ctx is None:
            ctx = Context.current()
        super().__init__(struct)
        self.ctx = ctx
        if name:
            self.set_name(name)

    @staticmethod
    def empty(size: list[int], dtype: DType, name: str = None) -> 'Tensor':
        size_arr = ffi.new("int64_t []", size)
        ret = Tensor(lib.ggml_new_tensor(Context.current(), dtype, len(size), size_arr), name)
        return ret

    @staticmethod
    def new(values: float | int | list[int | float], dtype: DType, name: str = None):

        assert dtype.is_int() or dtype.is_float(), "DType not supported."
        match values:
            case float(f) if dtype == DType.f32:
                raw_ptr = lib.ggml_new_f32(Context.current(), f)
            case int(i) if dtype == DType.i32:
                raw_ptr = lib.ggml_new_i32(Context.current(), i)
            case list(vals) if dtype in (DType.f32, DType.i32):
                ret = Tensor.empty([len(vals)], dtype, name)
                data_ptr = dtype.ffi_cast(ret.struct.data)
                data_ptr[0:len(vals)] = vals
                return ret
            case _:
                raise Exception("Not supported.")
        return Tensor(raw_ptr, name)

    def as_ndarray(self) -> np.ndarray:
        buffer = ffi.buffer(self.struct.data, size=prod(self.shape) * self.dtype.size)
        return np.frombuffer(buffer, dtype=(self.dtype.as_numpy_type())).reshape(self.shape)

    @property
    def name(self):
        return ffi.string(lib.ggml_get_name(self)).decode()

    @name.setter
    def name(self, name: str):
        lib.ggml_set_name(self, c_str(name))

    def set_name(self, name: str) -> 'Tensor':
        """Uses the name prefixes."""
        self.name = "".join(name_prefixes) + name
        return self

    @property
    def dtype(self) -> DType:
        return DType(self.struct.type)

    @property
    def grad(self) -> Optional['Tensor']:
        t = self.struct.grad
        if t:
            return Tensor(t, ctx=self.ctx)

    def set_param(self):
        lib.ggml_set_param(self.ctx, self)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(lib.ggml_mul(self.ctx, self, other), ctx=self.ctx)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(lib.ggml_add(self.ctx, self, other), ctx=self.ctx)

    def __matmul__(self, other) -> 'Tensor':
        return Tensor(lib.ggml_mul_mat(Context.current(), self, other))

    def set(self, val):
        match val:
            case float(f):
                lib.ggml_set_f32(self, f)
            case int(i):
                lib.ggml_set_i32(self, i)
            case _:
                raise Exception("Dunno what to do with that type yet. Pls implement.")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(self.struct.ne)

    def rms_norm(self, mean_offset: float, name: str = None) -> 'Tensor':
        """Root-mean-square normalization with an offset to the computed mean."""
        return Tensor(lib.ggml_rms_norm(Context.current(), self, mean_offset), name)

    def repeat(self, other: 'Tensor', name: str = None) -> 'Tensor':
        """Repeat tensor to fit into another tensor."""
        return Tensor(lib.ggml_repeat(Context.current(), self, other), name)

    def reshape(self, shape: list[int], name: str = None) -> 'Tensor':
        assert 0 < len(shape) <= 4
        if shape.count(-1):
            assert shape.count(-1) == 1, "Can only have a single inferred dimension."
            missing_dim = -prod(self.shape) // prod(shape)
            assert -missing_dim * prod(shape) == prod(self.shape), "Cannot find dimension to make the shape fit."
            shape[shape.index(-1)] = missing_dim

        struct = getattr(lib, f"ggml_reshape_{len(shape)}d")(Context.current(), self, *shape)
        return Tensor(struct, name)

    def permute(self, axis_order: list[int], name: str = None) -> 'Tensor':
        assert list(range(len(axis_order))) == sorted(axis_order), "Bad reordering."
        axis_order += list(range(len(axis_order), 4))  # fill in missing indices
        return Tensor(lib.ggml_permute(Context.current(), self, *axis_order), name)

    def contiguous(self, name: str = None) -> 'Tensor':
        return Tensor(lib.ggml_cont(Context.current(), self), name)

    @staticmethod
    def rope_custom(a: 'Tensor', indices: 'Tensor', n_dims: int, mode: int, n_ctx, n_orig_ctx: int, freq_base: float,
                    freq_scale: float, ext_factor: float, attn_factor: float, beta_fast: float, beta_slow: float,
                    name: str = None):
        raw_ptr = lib.ggml_rope_custom(Context.current(), a, indices, n_dims, mode, n_ctx, n_orig_ctx, freq_base,
                                       freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
        return Tensor(raw_ptr, name)

    @staticmethod
    def flash_attention(queries: 'Tensor', keys: 'Tensor', values: 'Tensor', masked=True, name: str = None) -> 'Tensor':
        raw_ptr = lib.ggml_flash_attn(Context.current(), queries, keys, values, masked)
        return Tensor(raw_ptr, name)

    def silu(self, name: str = None) -> 'Tensor':
        return Tensor(lib.ggml_silu(Context.current(), self), name)

    def get_rows(self, idxs: 'Tensor', name: str = None) -> 'Tensor':
        return Tensor(lib.ggml_get_rows(Context.current(), self, idxs), name)

    def __getitem__(self, item: int | tuple):
        # TODO check the order of the indices, I might've gotten them wrong
        if not isinstance(item, tuple):
            item = (item,)

        match item:
            case ints if all(isinstance(i, int) for i in ints):
                # insert 0s up to 4 dimensions because the ggml_get functions are hardcoded to 4 dimensions
                item += (4 - len(item)) * (0,)
                if self.dtype.is_float():
                    return lib.ggml_get_f32_nd(self, *item)
                elif self.dtype.is_int():
                    return lib.ggml_get_i32_nd(self, *item)
                else:
                    raise Exception(f"Cannot convert type {self.dtype.name} to a python-readable format.")

            case (Tensor() as t, *rest):
                assert all(r == slice(None, None, None) for r in rest), "This indexing is not yet supported."
                assert t.dtype == DType.i32, "Can only slice using i32 tensors."
                raw_ptr = lib.ggml_get_rows(Context.current(), self, t)
                return Tensor(raw_ptr)

    def __str__(self):
        return f"Tensor(shape={str(self.shape)+',':<21} type={str(self.dtype) + ',':<5} name='{self.name}')"

    def __repr__(self):
        return str(self)
