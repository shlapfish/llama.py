from llama import Model, Context, Sequence, Mirostatv2Sampler
from bindings import load_libllama, free_backend

# 1. git clone llama.cpp / PowerInfer
# 2. build the libllama.so target (see repo info)
# 3. point load_libllama() to the repo. We expect libllama.so to be in the root folder.

load_libllama("../llama.cpp")
# load_libllama("../PowerInfer")


model = Model(path="models/nous-hermes-llama2-13b.Q5_K_M.gguf", n_gpu_layers=0)

context = Context(model, n_ctx=2084, n_batch=1, threads=12)

txt = f"""### Instruction:
You are a helpful online assistant. You will answer any questions in detail to the best of your knowledge.
### User:
Why are bananas curved?
### Assistant:
"""

print(txt, end="")
seq = Sequence(context)
seq.insert(model.token_bos, generate_logits=False)
logits = seq.insert(txt)
context.process()

sampler = Mirostatv2Sampler(context, target_entropy=0)

for _ in range(512):
    new_token = sampler.sample(logits)
    if new_token == model.token_eos:
        break
    print(model.detokenize(new_token), end="")
    logits = seq.insert(new_token)
    context.process()

context.free()
model.free()
free_backend()
