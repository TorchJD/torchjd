from collections import deque
from typing import Set

import torch
from torch import Node, Tensor

from ._utils import append_to_dict, get_jacobian_and_next_nodes


def jac(starting_node: Node, inputs: Set[Tensor]) -> dict[Tensor, Tensor]:
    result = {}
    jacs = deque([(starting_node, torch.ones_like(starting_node).diag())])
    while jacs:
        curr_node, curr_jac = jacs.pop()
        if curr_node.__class__.__name__ == "AccumulateGrad":
            if curr_node.variable in inputs:
                result = append_to_dict(result, curr_node.variable, curr_jac)
        else:
            jacobian, next_functions = get_jacobian_and_next_nodes(curr_node)
            next_jacs = torch.vmap(jacobian)(curr_jac)
            next_couples = [
                (next_function, next_jac)
                for next_function, next_jac in zip(next_functions, next_jacs)
            ]
            jacs.extend(next_couples)
    return result
