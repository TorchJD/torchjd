from collections.abc import Callable

import torch
from torch import Tensor


def get_output_and_gramian(func: Callable, *primals) -> tuple[Tensor, Tensor]:
    output, vjp_fn = torch.func.vjp(func, *primals)

    def vgp_fn(v: Tensor) -> Tensor:
        return torch.func.jvp(func, primals, tangents=vjp_fn(v))[1]

    identity = torch.eye(output.shape[0])
    gramian = torch.func.vmap(vgp_fn)(identity)

    return output, gramian
