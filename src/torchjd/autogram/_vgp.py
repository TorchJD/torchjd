from collections.abc import Callable

import torch
from torch import Tensor


def vgp(func: Callable, *primals) -> tuple[Tensor, Callable]:
    output, vjp_fn = torch.func.vjp(func, *primals)

    def vgp_fn(v: Tensor) -> Tensor:
        return torch.func.jvp(func, primals, tangents=vjp_fn(v))[1]

    return output, vgp_fn
