from src.bindings import free_backend
from llama import Model, Context, Sequence
from bindings import load_libllama
from src.sampling import Mirostatv2Sampler

load_libllama("../llama.cpp")


model = Model(path="models/nous-hermes-llama2-13b.Q4_0.gguf", n_gpu_layers=0)

context = Context(model, n_ctx=512, n_batch=16)

txt = f"### Instruction:\nHow do I get rich fast?\n### Response:\n"
assert txt == model.detokenize(model.tokenize(txt))

print(txt, end="")
seq = Sequence(context)
seq.insert(model.token_bos, generate_logits=False)
logits = seq.insert(txt)
context.process()

sampler = Mirostatv2Sampler(context)

for _ in range(512):
    new_token = sampler.sample(logits)
    print(model.detokenize(new_token), end="")
    logits = seq.insert(new_token)
    context.process()

context.free()
model.free()
free_backend()
