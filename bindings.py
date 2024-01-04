import pathlib
import re
from dataclasses import dataclass
from typing import Any

import cffi

ffi = cffi.FFI()


@dataclass(slots=True)
class StructWrapper:
    """Marker class for wrapping structs. Makes the class usable as an argument to lib functions."""
    struct: Any
    """cffi pointer to the raw struct."""


class Lib:
    def __init__(self):
        self.lib = None

    def __getattr__(self, item):
        if self.lib is None:
            raise Exception("do bindings.load_libllama(path) before other imports")
        o = getattr(self.lib, item)
        if isinstance(o, (int, float)):
            return o
        else:
            # convert wrapped arguments and call function
            return lambda *args: o(*(a.struct if isinstance(a, StructWrapper) else a for a in args))


lib = Lib()


def load_libllama(path: str, numa=False):
    """Cleans up llama.h and then uses it to generate bindings."""
    path = pathlib.Path(path).absolute()
    h_file = (path / "ggml.h").read_text()
    h_file += "\n" + (path / "llama.h").read_text()

    h_file = h_file.replace("sizeof(int32_t)", "4")

    # for PowerInfer
    h_file = h_file.replace("atomic_int", "int")

    to_remove = [
        re.compile(r"\\\n", re.MULTILINE),
        re.compile(r'#ifdef +__cplusplus\nextern "C" \{\n#endif'),
        re.compile(r"#ifdef +__cplusplus\n}\n#endif"),
        re.compile(r"#(?!define [A-Z_]+ +[-0-9]+).*"),
        re.compile(r"typedef (half|__fp16) ggml_fp16_t;"),
        re.compile(r"\n[^\n]*LLAMA_API DEPRECATED[^;]*;"),
        re.compile(r"GGML_DEPRECATED[^;]*;"),
        re.compile(r"\n[^\n]*llama_internal_get_tensor_map[^;]*;"),
        re.compile(r"\n[^\n]*ggml_log_callback[^;]*;"),
        re.compile(r"(LLAMA|GGML)_API"),
        re.compile(r"GGML_ATTRIBUTE_FORMAT.*"),
        re.compile(r"GGML_RESTRICT"),

        # for PowerInfer
        re.compile(r".*using std::.*")
    ]
    for p in to_remove:
        h_file = p.sub("", h_file)
    print(h_file)
    ffi.cdef(h_file)
    object.__setattr__(lib, "lib", ffi.dlopen((path / "libllama.so").as_posix()))
    lib.llama_backend_init(numa)


def free_backend():
    lib.llama_backend_free()


def c_str(s: str):
    return ffi.new("char[]", s.encode())


def copy_array(src, length, type_str):
    ret = ffi.new(type_str + "[]", length)
    size = ffi.sizeof(type_str)
    ffi.memmove(ret, src, size * length)
    return ret


class UseAfterFree(Exception):
    pass