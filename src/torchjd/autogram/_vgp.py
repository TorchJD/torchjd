from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def vgp(
    func: Callable, *primals, has_aux: bool = False
) -> tuple[Tensor, Callable[[Tensor], Tensor]] | tuple[Tensor, Callable[[Tensor], Tensor], Any]:
    if has_aux:
        output, vjp_fn, aux = torch.func.vjp(func, *primals, has_aux=True)
    else:
        output, vjp_fn = torch.func.vjp(func, *primals, has_aux=False)
        aux = None

    if output.ndim != 1:
        raise ValueError("The function should return a vector")

    def vgp_fn(v: Tensor) -> Tensor:
        return torch.func.jvp(func, primals, tangents=vjp_fn(v))[1]

    if has_aux:
        return output, vgp_fn, aux
    else:
        return output, vgp_fn


def get_gramian(vgp_fn: Callable[[Tensor], Tensor], m: int) -> Tensor:
    identity = torch.eye(m)
    gramian = torch.func.vmap(vgp_fn)(identity)

    return gramian


def get_output_and_gramian(func: Callable, *primals) -> tuple[Tensor, Tensor]:
    output, vgp_fn = vgp(func, *primals)
    gramian = get_gramian(vgp_fn, output.shape[0])

    return output, gramian
