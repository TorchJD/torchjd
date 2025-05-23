from collections import deque
from typing import Set

import torch
from torch import Tensor


def jac(output: Tensor, inputs: Set[Tensor]) -> dict[Tensor, Tensor]:
    result = {}
    jacs = deque([(output.grad_fn, torch.ones_like(output).diag())])
    while jacs:
        curr_node, curr_jac = jacs.pop()
        if curr_node.__class__.__name__ == "AccumulateGrad":
            if curr_node.variable in inputs:
                result[curr_node.variable] = curr_jac
        else:
            next_functions = [fn[0] for fn in curr_node.next_functions]
            next_functions, indices = zip(
                *[(fn, i) for i, fn in enumerate(next_functions) if fn is not None]
            )

            def get_vjp(grad: Tensor) -> tuple[Tensor, ...]:
                output = [t for i, t in enumerate(curr_node(grad)) if i in indices]
                return tuple(output)

            next_jacs = torch.vmap(get_vjp)(curr_jac)
            next_couples = [
                (next_function, next_jac)
                for next_function, next_jac in zip(next_functions, next_jacs)
                if next_function is not None
            ]
            jacs.extend(next_couples)
    return result
