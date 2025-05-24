from collections import deque
from typing import Set

import torch
from torch import Tensor

from ._utils import accumulate_to_gramian, get_jacobian_and_next_nodes


def gram(output: Tensor, inputs: Set[Tensor]) -> Tensor:
    m = output.shape[0]
    result = torch.zeros([m, m], device=output.device, dtype=output.dtype)
    jacs = deque([(output.grad_fn, torch.ones_like(output).diag())])
    while jacs:
        curr_node, curr_jac = jacs.pop()
        if curr_node.__class__.__name__ == "AccumulateGrad":
            if curr_node.variable in inputs:
                result = accumulate_to_gramian(result, curr_node.variable, curr_jac)
        else:
            jacobian, next_functions = get_jacobian_and_next_nodes(curr_node)
            next_jacs = torch.vmap(jacobian)(curr_jac)
            next_couples = [
                (next_function, next_jac)
                for next_function, next_jac in zip(next_functions, next_jacs)
            ]
            jacs.extend(next_couples)
    return result
