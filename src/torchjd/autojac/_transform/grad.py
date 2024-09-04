from typing import Iterable, Sequence

import torch
from torch import Tensor

from ._differentiate import _Differentiate
from ._utils import _materialize
from .tensor_dict import Gradients


class Grad(_Differentiate[Gradients]):
    def __init__(
        self,
        outputs: Iterable[Tensor],
        inputs: Iterable[Tensor],
        retain_graph: bool = False,
        create_graph: bool = False,
    ):
        super().__init__(outputs, inputs, retain_graph, create_graph)

    def _differentiate(self, grad_outputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        return _grad(
            outputs=list(self.outputs),
            inputs=list(self.inputs),
            grad_outputs=grad_outputs,
            retain_graph=self.retain_graph,
            create_graph=self.create_graph,
            allow_unused=True,
        )


def _grad(
    outputs: Sequence[Tensor],
    inputs: Sequence[Tensor],
    grad_outputs: Sequence[Tensor],
    retain_graph: bool,
    create_graph: bool,
    allow_unused: bool,
) -> tuple[Tensor, ...]:
    """
    Wraps `autograd.grad` to give it additional responsibilities that it should have (like being
    able to work with an empty sequence of `inputs`).
    """

    if len(inputs) == 0:
        return tuple()

    if len(outputs) == 0:
        return tuple(
            [torch.empty(input.shape, device=input.device, dtype=input.dtype) for input in inputs]
        )

    optional_grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )
    grads = _materialize(optional_grads, inputs)
    return grads
