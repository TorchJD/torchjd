from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from torch import Tensor

from ._utils import ordered_set
from .base import _A, Transform


class _Differentiate(Transform[_A, _A], ABC):
    def __init__(
        self,
        outputs: Iterable[Tensor],
        inputs: Iterable[Tensor],
        retain_graph: bool,
        create_graph: bool,
    ):
        self.outputs = ordered_set(outputs)
        self.inputs = ordered_set(inputs)
        self.retain_graph = retain_graph
        self.create_graph = create_graph

    def _compute(self, tensors: _A) -> _A:
        tensor_outputs = [tensors[output] for output in self.outputs]

        differentiated_tuple = self._differentiate(tensor_outputs)
        new_differentiations = dict(zip(self.inputs, differentiated_tuple))
        return type(tensors)(new_differentiations)

    @abstractmethod
    def _differentiate(self, tensor_outputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        raise NotImplementedError

    @property
    def required_keys(self) -> set[Tensor]:
        # outputs in the forward direction become inputs in the backward direction, and vice-versa
        return set(self.outputs)

    @property
    def output_keys(self) -> set[Tensor]:
        # outputs in the forward direction become inputs in the backward direction, and vice-versa
        return set(self.inputs)
