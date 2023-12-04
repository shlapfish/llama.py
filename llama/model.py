from typing import Iterable

from bindings import c_str, lib, ffi, UseAfterFree


class Model:
    """
    - path: `str` --- Path to the .gguf to load.
    - gpu_layers: `int` --- Number of layers to offload to the GPU. They will be stored in VRAM only.
    - main_gpu: `int` --- Index of the GPU to use for scratch and small tensors. Get a list from #todo
    - vocab_only: `bool` --- Only load the vocabulary, no weights.
    - use_mmap: `bool` --- Use memory mapping to load the model.
    - use_mlock: `bool` --- Force system to keep model in RAM.
    """
    def __init__(self, path: str, **kwargs):
        path = c_str(path)
        params = lib.llama_model_default_params()
        for k, v in kwargs.items():
            setattr(params, k, v)
        self._raw = lib.llama_load_model_from_file(path, params)
        if self._raw == ffi.NULL:
            raise Exception("Could not load model.")
        self.token_bos: int = lib.llama_token_bos(self._raw)
        """Beginning of sequence token."""

        self.token_eos: int = lib.llama_token_eos(self._raw)
        """End of sequence token."""

        self.token_newline: int = lib.llama_token_nl(self._raw)
        """Newline token."""

        self.token_prefix: int = lib.llama_token_prefix(self._raw)
        """Beginning of infill prefix (codellama)."""

        self.token_middle: int = lib.llama_token_middle(self._raw)
        """Beginning of infill middle (codellama)."""

        self.token_suffix: int = lib.llama_token_suffix(self._raw)
        """Beginning of infill suffix (codellama)."""

        self.token_eot: int = lib.llama_token_eot(self._raw)
        """End of infill middle (codellama)."""

    def tokenize(self, text: str, tokenize_specials=True) -> list[int]:
        """
        :param tokenize_specials: Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.
        :param add_bos: Add a beginning-of-sequence token.
        """
        text = c_str(text)
        tokens = ffi.new("llama_token[]", len(text))
        tokens_created = lib.llama_tokenize(self._raw, text,
                                            len(text) - 1,  # don't include null byte
                                            tokens, len(text), False, tokenize_specials)
        if tokens_created < 0:
            raise Exception("Could not tokenize string.")
        return list(tokens[0:tokens_created])

    def detokenize(self, tokens: int | list[int]) -> str:
        if not isinstance(tokens, Iterable):
            tokens = [tokens]
        buf = ffi.new("char[16]")
        result = []
        for token in tokens:
            n_chars = lib.llama_token_to_piece(self._raw, token, buf, len(buf))
            try:
                result.append(ffi.string(buf, n_chars).decode())
            except UnicodeDecodeError: pass
        return "".join(result)

    @property
    def description(self) -> str:
        """A string description of the model type."""
        buf = ffi.new("char[1024]")
        size = lib.llama_model_desc(self._raw, buf, len(buf))
        return ffi.string(buf, size).decode()

    @property
    def total_size(self) -> int:
        """The total size of all the tensors in the model in bytes."""
        return lib.llama_model_size(self._raw)

    @property
    def num_parameters(self) -> int:
        """The total number of parameters in the model."""
        return lib.llama_model_n_params(self._raw)

    @property
    def vocab_size(self) -> int:
        """The amount of tokens in the vocabulary."""
        return lib.llama_n_vocab(self._raw)

    @property
    def training_context_size(self) -> int:
        """The context size on which the model was trained."""
        return lib.llama_n_ctx_train(self._raw)

    @property
    def embedding_size(self) -> int:
        return lib.llama_n_embd(self._raw)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()

    def free(self):
        """Unloads the model, making it unusable."""
        if self._raw is not None:
            lib.llama_free_model(self._raw)
        self._raw = None

    def __del__(self):
        try:
            if self._raw is not None:
                self.free()
        except UseAfterFree:
            pass