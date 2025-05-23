from collections import deque
from typing import Set

import torch
from torch import Tensor


def grad(output: Tensor, inputs: Set[Tensor]) -> dict[Tensor, Tensor]:
    result = {}
    grads = deque([(output.grad_fn, torch.ones_like(output))])
    while grads:
        curr_node, curr_grad = grads.pop()
        if curr_node.__class__.__name__ == "AccumulateGrad":
            if curr_node.variable in inputs:
                result[curr_node.variable] = curr_grad
        else:
            next_functions = [fn[0] for fn in curr_node.next_functions]
            next_grads = curr_node(curr_grad)
            next_couples = [
                (next_function, next_grad)
                for next_function, next_grad in zip(next_functions, next_grads)
                if next_function is not None
            ]
            grads.extend(next_couples)
    return result
