from bindings import lib, load_libllama
load_libllama("../../llama.cpp")

from ggml import Context, Tensor, DType, Graph


lib.ggml_time_init()


class Gguf:
    def __int__(self, raw_ptr):
        self._raw = raw_ptr


MiB = 1024 * 1024
with Context.new(128 * MiB) as ctx:
    x = Tensor.new([1], DType.f32, "x")
    x.set_param()
    a = Tensor.new([1], DType.f32, "a")
    b = Tensor.new([1], DType.f32, "b")

    f = a * x * x + b

    forward = Graph(grads=True)
    forward.build_forward_expand(f)
    backward = Graph(grads=True)
    lib.ggml_build_backward_expand(ctx, forward, backward, False)

    # set the input variable and parameter values
    x.set(2.)
    a.set(3.)
    b.set(4.)

    f.grad.set(1.)
    forward.compute()
    backward.compute()
    print(f"x' = {x.grad[0]}")
    assert f[0] == 16.




    backward.compute()

    forward.print()
    backward.print()
