from collections import deque
from typing import Set

import torch
from torch import Tensor

from ._utils import append_to_dict, get_jacobian_and_next_nodes


def grad(output: Tensor, inputs: Set[Tensor]) -> dict[Tensor, Tensor]:
    result = {}
    grads = deque([(output.grad_fn, torch.ones_like(output))])
    while grads:
        curr_node, curr_grad = grads.pop()
        if curr_node.__class__.__name__ == "AccumulateGrad":
            if curr_node.variable in inputs:
                result = append_to_dict(result, curr_node.variable, curr_grad)
        else:
            jacobian, next_functions = get_jacobian_and_next_nodes(curr_node)
            next_grads = jacobian(curr_grad)
            next_couples = [
                (next_function, next_grad)
                for next_function, next_grad in zip(next_functions, next_grads)
            ]
            grads.extend(next_couples)
    return result
