from typing import Callable

import torch
from torch import Node, Tensor


def get_jacobian_and_next_nodes(
    node: Node,
) -> tuple[Callable[[Tensor], tuple[Tensor, ...]], list[Node]]:
    next_functions = [fn[0] for fn in node.next_functions]
    next_functions, indices = zip(
        *[(fn, i) for i, fn in enumerate(next_functions) if fn is not None]
    )

    def jacobian(grad: Tensor) -> tuple[Tensor, ...]:
        output = [t for i, t in enumerate(node(grad)) if i in indices]
        return tuple(output)

    return jacobian, next_functions


def append_to_dict(
    tensors: dict[Tensor, Tensor], tensor: Tensor, derivative: Tensor
) -> dict[Tensor, Tensor]:
    tensors[tensor] = tensor
    return tensors


def accumulate_to_gramian(gramian: Tensor, _: Tensor, jacobian: Tensor) -> Tensor:
    reshaped_jac = jacobian.reshape([gramian.shape[0], -1])
    return torch.addmm(gramian, reshaped_jac, reshaped_jac.T)
