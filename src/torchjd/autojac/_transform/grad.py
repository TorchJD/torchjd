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
        outputs = list(self.outputs)
        inputs = list(self.inputs)

        if len(inputs) == 0:
            return tuple()

        if len(outputs) == 0:
            return tuple(
                [
                    torch.empty(input.shape, device=input.device, dtype=input.dtype)
                    for input in inputs
                ]
            )

        optional_grads = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=self.retain_graph,
            create_graph=self.create_graph,
            allow_unused=True,
        )
        grads = _materialize(optional_grads, inputs)
        return grads
