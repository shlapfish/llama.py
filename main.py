from src.bindings import free_backend
from src.llama import Model, Context, Sequence, lib, load_libllama, initialize_backend
import numpy as np

load_libllama("/your/github/clone/of/llama.cpp")
initialize_backend(numa=False)
model = Model(path="/your/model.gguf",
              n_gpu_layers=0)

context = Context(model, n_ctx=2048, n_batch=16)

txt = model.bos + "What are the different types of footwear?" + model.nl
assert txt == model.detokenize(model.tokenize(txt))

print(txt, end="")
seq = Sequence(context)
logits = seq.insert(txt, generate_logits=True)
context.process()

for _ in range(100):
    lst = logits.get()
    lst[:4] = 4*[0.]
    new_token = int(np.argmax(lst))
    print(model.detokenize(new_token), end="")
    logits = seq.insert(new_token, generate_logits=True)
    context.process()

context.free()
model.free()
free_backend()
