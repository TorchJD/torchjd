from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from torch import Tensor

from torchjd._transform._utils import ordered_set
from torchjd._transform.base import _A, Transform


class _Differentiation(Transform[_A, _A], ABC):
    def __init__(self, outputs: Iterable[Tensor], inputs: Iterable[Tensor]):
        self.outputs = ordered_set(outputs)
        self.inputs = ordered_set(inputs)

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
