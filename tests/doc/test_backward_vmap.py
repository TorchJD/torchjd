import torch
from torch import Tensor, vmap


class backward_fn(torch.autograd.Function):
    @staticmethod
    def forward(t: Tensor) -> None:
        return None

    @staticmethod
    def setup_context(*_):
        pass

    @staticmethod
    def vmap(info, in_dims: tuple[int | None], t: Tensor) -> tuple[None, None]:
        return None, None


class test_fn(torch.autograd.Function):

    @staticmethod
    def forward(t: Tensor) -> Tensor:
        return t

    @staticmethod
    def setup_context(*_):
        pass

    @staticmethod
    def backward(ctx, t: Tensor) -> Tensor:
        backward_fn.apply(t)
        return t


def test_backward_vmap():
    a = torch.randn([3], requires_grad=True)
    b = torch.randn([3])

    c = test_fn.apply(a + b)

    def differentiation(grads):
        return torch.autograd.grad(c, a, grad_outputs=grads)

    vmap(differentiation)(torch.diag(torch.ones_like(c)))
