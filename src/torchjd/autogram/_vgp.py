from collections.abc import Callable

import torch
from torch import Tensor


def get_output_and_gramian(func: Callable, *primals) -> tuple[Tensor, Tensor]:
    output, vjp_fn = torch.func.vjp(func, *primals)

    def vgp_fn(v: Tensor) -> Tensor:
        return torch.func.jvp(func, primals, tangents=vjp_fn(v))[1]

    columns = []
    for e in torch.eye(output.shape[0]):
        columns.append(vgp_fn(e).unsqueeze(1))
    gramian = torch.hstack(columns)

    return output, gramian
