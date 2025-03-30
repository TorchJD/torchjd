from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from torch import Tensor

from .base import _A, Transform
from .ordered_set import ordered_set


class _Differentiate(Transform[_A, _A], ABC):
    def __init__(
        self,
        outputs: Iterable[Tensor],
        inputs: Iterable[Tensor],
        retain_graph: bool,
        create_graph: bool,
    ):
        self.outputs = list(outputs)
        self.inputs = ordered_set(inputs)
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def __call__(self, tensors: _A) -> _A:
        tensor_outputs = [tensors[output] for output in self.outputs]

        differentiated_tuple = self._differentiate(tensor_outputs)
        new_differentiations = dict(zip(self.inputs, differentiated_tuple))
        return type(tensors)(new_differentiations)

    @abstractmethod
    def _differentiate(self, tensor_outputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """
        Abstract method for differentiating the outputs with respect to the inputs, and applying the
        linear transformations represented by the tensor_outputs to the results.

        The implementation of this method should define what kind of differentiation is performed:
        whether gradients, Jacobians, etc. are computed, and what the dimension of the
        tensor_outputs should be.
        """

    def check_and_get_keys(self) -> tuple[set[Tensor], set[Tensor]]:
        # outputs in the forward direction become inputs in the backward direction, and vice-versa
        return set(self.outputs), set(self.inputs)
