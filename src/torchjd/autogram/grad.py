import torch
from torch import Tensor
from torch.autograd.graph import get_gradient_edge

from .differentiation_graph import Derivatives, get_jacobian_and_children, topological_sort


def grad(outputs: list[Tensor], inputs: set[Tensor]) -> dict[Tensor, Tensor]:
    result = {}
    grads = Derivatives([(output, torch.ones_like(output)) for output in outputs])
    roots = [get_gradient_edge(output)[0] for output in outputs]
    leaves = {get_gradient_edge(input)[0]: input for input in inputs}
    nodes = topological_sort(roots, set(leaves.keys()), set())
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
