from collections.abc import Callable

import torch
from torch import Tensor


def vgp(func: Callable, *primals) -> tuple[Tensor, Callable]:
    output, vjp_fn = torch.func.vjp(func, *primals)

    def vgp_fn(grad_output: Tensor) -> Tensor:
        grads = torch.autograd.grad(
            output,
            list(primals),
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        return vjp_fn(*grads)[0]

    return output, vgp_fn
