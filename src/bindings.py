import pathlib
import re

import cffi

ffi = cffi.FFI()


class Lib:
    def __init__(self):
        self.lib = None

    def __getattr__(self, item):
        if self.lib is None:
            raise Exception("Load llama.cpp library first with load_libllama(path)")
        return getattr(self.lib, item)


lib = Lib()


def load_libllama(path: str):
    """Cleans up llama.h and then uses it to generate bindings."""
    path = pathlib.Path(path).absolute()
    h_file = (path / "llama.h").read_text()

    to_remove = [
        re.compile(r"\n[^\n]*LLAMA_API DEPRECATED[^;]*;"),
        re.compile(r"\n[^\n]*llama_internal_get_tensor_map[^;]*;"),
        re.compile(r'#ifdef __cplusplus\nextern "C" \{\n#endif'),
        re.compile(r"#ifdef __cplusplus\n}\n#endif"),
        re.compile(r"\n[^\n]*ggml_log_callback[^;]*;"),
        re.compile(r"LLAMA_API"),
        re.compile(r"^#[^\n]*\n", re.MULTILINE),
    ]
    for p in to_remove:
        h_file = p.sub("", h_file)

    ffi.cdef(h_file)
    object.__setattr__(lib, "lib", ffi.dlopen((path / "libllama.so").as_posix()))


backend_initialized = False


def initialize_backend(numa: bool = False):
    global backend_initialized
    if backend_initialized:
        raise Exception("The backend is already initialized.")
    lib.llama_backend_init(numa)
    backend_initialized = True


def free_backend():
    global backend_initialized
    if not backend_initialized:
        lib.llama_backend_free()
    backend_initialized = False


def c_str(s: str):
    return ffi.new("char[]", s.encode())


def copy_array(src, length, type_str):
    ret = ffi.new(type_str + "[]", length)
    size = ffi.sizeof(type_str)
    ffi.memmove(ret, src, size * length)
    return ret