import torch
from torch import Tensor

from .differentiation_graph import (
    Derivatives,
    get_jacobian_and_children,
    get_node,
    topological_sort,
)


def grad(outputs: list[Tensor], inputs: set[Tensor], excluded: set[Tensor]) -> dict[Tensor, Tensor]:
    result = {}
    grads = Derivatives([(output, torch.ones_like(output)) for output in outputs])
    roots = [get_node(output) for output in outputs]
    leaves = {get_node(input): input for input in inputs}
    excluded = {get_node(tensor) for tensor in excluded}
    nodes = topological_sort(roots, set(leaves.keys()), excluded)
    for node in nodes:
        node_grad = grads.pop(node)
        if node in leaves:
            t = leaves[node]
            result[t] = node_grad
        else:
            jacobian, next_functions = get_jacobian_and_children(node)
            next_grads = jacobian(node_grad)
            grads.update(zip(next_functions, next_grads))
    return result
