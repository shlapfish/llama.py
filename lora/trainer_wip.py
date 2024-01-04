import numpy as np

from bindings import lib, load_libllama
load_libllama("../llama.cpp")
from llama import Model


from ggml import Context, Tensor, DType, Gguf, name_prefix, Graph

lib.ggml_time_init()


class LlamaBlock:
    def __init__(self, gguf: Gguf, idx: int):
        self.name = f"blk.{idx}"
        self.idx = idx
        self.gguf = gguf
        self.attn_norm = self.get("attn_norm")
        self.ffn_norm = self.get("ffn_norm")
        self.attn_q = self.get("attn_q")  # wq
        self.attn_k = self.get("attn_k")  # wk
        self.attn_v = self.get("attn_v")  # wv
        self.attn_output = self.get("attn_output")  # wo
        self.ffn_gate = self.get("ffn_gate")  # w1
        self.ffn_down = self.get("ffn_down")  # w2
        self.ffn_up = self.get("ffn_up")  # w3
        self.layer_norm_rms_epsilon = gguf.info["llama.attention.layer_norm_rms_epsilon"]
        self.n_heads = gguf.info["llama.attention.head_count"]
        self.n_heads_kv = gguf.info["llama.attention.head_count_kv"]
        self.n_context = gguf.info["llama.context_length"]
        self.rope_freq_base = 10_000.  # doesn't seem to be in my llama guff
        self.rope_freq_scale = 1.

    def get(self, name: str):
        return self.gguf.tensors[f"{self.name}.{name}.weight"]

    def rope(self, t: Tensor, indices: Tensor, name: str = None) -> Tensor:
        return Tensor.rope_custom(a=t, indices=indices, n_dims=t.shape[0], mode=0, n_ctx=self.n_context,
                                  n_orig_ctx=0, freq_base=self.rope_freq_base, freq_scale=self.rope_freq_scale,
                                  ext_factor=0., attn_factor=1., beta_fast=0., beta_slow=0., name=name)

    def apply_attention_layer(self, input_embedding: Tensor, embedding_indices: Tensor) -> Tensor:
        n_ctx = embedding_indices.shape[0]
        n_embd = input_embedding.shape[0]
        n_batch = input_embedding.shape[1] // n_ctx
        with name_prefix(self.name + ".attn."):
            normalized_embedding = input_embedding.rms_norm(self.layer_norm_rms_epsilon, name="normalized_embedding")
            scale_factor = self.attn_norm.repeat(normalized_embedding, name="scale_factor")
            weighted_embedding = (scale_factor * normalized_embedding).set_name("weighted_embedding")

            queries = self.attn_q @ weighted_embedding
            queries = queries.reshape([-1, self.n_heads, n_ctx, n_batch], name="pre_roped_queries")
            queries = self.rope(queries, embedding_indices, name="roped_queries")
            queries = queries.permute([0, 2, 1, 3], name="queries")

            keys = self.attn_k @ weighted_embedding
            keys = keys.reshape([-1, self.n_heads_kv, n_ctx, n_batch], name="pre_roped_keys")
            keys = self.rope(keys, embedding_indices, name="roped_keys")
            keys = keys.permute([0, 2, 1, 3], name="keys")

            values = self.attn_v @ weighted_embedding
            values = values.reshape([n_ctx, n_batch, -1, self.n_heads_kv], name="values")
            values = values.permute([0, 3, 1, 2], name="values_permuted")

            flash_attention = Tensor.flash_attention(queries, keys, values, masked=True, name="flash_attention")

            dense_input = flash_attention.permute([0, 2, 1, 3]).contiguous().reshape([n_embd, n_ctx * n_batch], name="dense_input")
            dense_output = (self.attn_output @ dense_input).set_name("dense_output")

            return (input_embedding + dense_output).set_name("output")

    def apply_feed_forward_layer(self, input_embedding: Tensor) -> Tensor:
        with name_prefix(self.name + ".ffl."):
            normalized_embedding = input_embedding.rms_norm(self.layer_norm_rms_epsilon, name="normalized_embedding")
            scale_factor = self.ffn_norm.repeat(normalized_embedding, name="scale_factor")
            weighted_embedding = (scale_factor * normalized_embedding).set_name(name="weighted_embedding")

            # first widen the embedding, then multiply by the gating and then narrow again to add to the embedding
            widened = (self.ffn_up @ weighted_embedding).set_name("widened")
            gate = (self.ffn_gate @ weighted_embedding).silu(name="gating")
            gated = (gate * widened).set_name("gated")
            narrowed = (self.ffn_down @ gated).set_name("narrowed")
            return (narrowed + input_embedding).set_name("output")


class LlamaGguf:
    def __init__(self, gguf: Gguf):
        self.n_blocks = gguf.info["llama.block_count"]
        self.n_embd = gguf.info["llama.embedding_length"]
        self.n_context = gguf.info["llama.context_length"]

        self.rope_dimension = gguf.info["llama.rope.dimension_count"]

        self.n_heads = gguf.info["llama.attention.head_count"]
        self.layer_norm_rms_epsilon = gguf.info["llama.attention.layer_norm_rms_epsilon"]

        self.token_embdeddings = gguf.tensors["token_embd.weight"]
        self.output_norm = gguf.tensors["output_norm.weight"]
        self.dense_output = gguf.tensors["output.weight"]

        self.blocks: list[LlamaBlock] = [LlamaBlock(gguf, i) for i in range(self.n_blocks)]

    def apply_model(self, input_tokens: Tensor) -> Tensor:
        assert input_tokens.dtype == DType.i32
        n_ctx = input_tokens.shape[0]
        n_batch = input_tokens.shape[1]
        # this is so weird, I would expect n_vocab to be the first shape[0]
        n_vocab = self.token_embdeddings.shape[1]
        rope_indices = Tensor.new(list(range(n_ctx)), DType.i32, name="rope_indices")

        # and here weird again, get_rows is actually t[:, idxs], which I would say is "get_columns"
        input_embedding = self.token_embdeddings.get_rows(input_tokens.reshape([-1])).set_name("input_embedding")
        current = input_embedding
        for block in self.blocks:
            current = block.apply_attention_layer(current, rope_indices)
            current = block.apply_feed_forward_layer(current)

        # the final output
        normalized_embedding = current.rms_norm(self.layer_norm_rms_epsilon, name="normalized")
        scale_factor = self.output_norm.repeat(normalized_embedding, name="scale_factor")
        weighted_embedding = (scale_factor * normalized_embedding).set_name("weighted_embedding")
        return (self.dense_output @ weighted_embedding).reshape([n_vocab, n_ctx, n_batch], name="output")


path = "models/llama-2-7b.Q4_K_S.gguf"
gguf = Gguf.load_from_file(path)
for k in gguf.info:
    try:
        print(f"{k} = {gguf.info[k]}")
    except:
        print(f"{k} = * READING TYPE NOT YET SUPPORTED *")
print("\n".join(str(t) for t in gguf.tensors))

llama = LlamaGguf(gguf)
MiB = 1024 * 1024
with Context.new(8000 * MiB) as ctx:

    model = Model(path, vocab_only=True)
    tokens = [model.token_bos] + model.tokenize("The five laws of magic are:\n")
    input_tokens = Tensor.new(tokens, DType.i32, "tokens_input")
    input_tokens.set_param()
    output = llama.apply_model(input_tokens)

    forward = Graph(grads=False)
    forward.build_forward_expand(output)
    forward.compute()
    arr = output.as_ndarray().squeeze((2, 3))
    print(f"{arr.dtype} {arr.shape}")
    expected_tokens = arr.argmax(axis=0)
    for tok in expected_tokens:
        print(model.detokenize(int(tok)))

