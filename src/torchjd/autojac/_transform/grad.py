from typing import Sequence

import torch
from torch import Tensor

from ._differentiate import Differentiate
from ._materialize import materialize
from .ordered_set import OrderedSet
from .tensor_dict import Gradients


class Grad(Differentiate[Gradients]):
    def __init__(
        self,
        outputs: OrderedSet[Tensor],
        inputs: OrderedSet[Tensor],
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

        if len(self.inputs) == 0:
            return tuple()

        if len(self.outputs) == 0:
            return tuple([torch.zeros_like(input) for input in self.inputs])

        optional_grads = torch.autograd.grad(
            self.outputs,
            self.inputs,
            grad_outputs=grad_outputs,
            retain_graph=self.retain_graph,
            create_graph=self.create_graph,
            allow_unused=True,
        )
        grads = materialize(optional_grads, self.inputs)
        return grads
