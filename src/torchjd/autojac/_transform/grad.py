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
        """
        Computes the gradient of each output with respect to each input, and applies the linear
        transformations represented by the grad_outputs to the results.

        Returns one gradient per input.

        :param grad_outputs: The sequence of scalar tensors to scale the obtained gradients with.
            Its length should be equal to the length of ``outputs``. Each grad_output should have
            the same shape as the corresponding output.
        """

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
