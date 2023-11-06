from src.bindings import free_backend
from src.llama import Model, Context, Sequence, lib, load_libllama, initialize_backend, ffi
from src.sampling import Mirostatv2Sampler

load_libllama("../llama.cpp")


initialize_backend(numa=False)
model = Model(path="models/Nous-Hermes-7B.Q4_K_M.gguf", n_gpu_layers=0)

context = Context(model, n_ctx=512, n_batch=16)

txt = f"{model.bos}### Instruction:{model.nl}How do I get rich fast?{model.nl}### Response:{model.nl}"
assert txt == model.detokenize(model.tokenize(txt))

print(txt, end="")
seq = Sequence(context)
logits = seq.insert(txt, generate_logits=True)
context.process()

sampler = Mirostatv2Sampler(context)

for _ in range(512):
    new_token = sampler.sample(logits)
    print(model.detokenize(new_token), end="")
    logits = seq.insert(new_token, generate_logits=True)
    context.process()

context.free()
model.free()
free_backend()
