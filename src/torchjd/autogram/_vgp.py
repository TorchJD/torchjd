from collections.abc import Callable

import torch
from torch import Tensor


def vgp(func: Callable, *primals) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    output, vjp_fn = torch.func.vjp(func, *primals)

    if output.ndim != 1:
        raise ValueError("The function should return a vector")

    def vgp_fn(v: Tensor) -> Tensor:
        return torch.func.jvp(func, primals, tangents=vjp_fn(v))[1]

    return output, vgp_fn


def get_gramian(vgp_fn: Callable[[Tensor], Tensor], m: int) -> Tensor:
    identity = torch.eye(m)
    gramian = torch.func.vmap(vgp_fn)(identity)

    return gramian


def get_output_and_gramian(func: Callable, *primals) -> tuple[Tensor, Tensor]:
    output, vgp_fn = vgp(func, *primals)
    gramian = get_gramian(vgp_fn, output.shape[0])

    return output, gramian
