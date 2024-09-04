from typing import Iterable

import torch
from torch import Tensor

from .base import Transform
from .tensor_dict import EmptyTensorDict, Gradients


class Init(Transform[EmptyTensorDict, Gradients]):
    def __init__(self, values: Iterable[Tensor]):
        self.values = set(values)

    def _compute(self, input: EmptyTensorDict) -> Gradients:
        r"""
        Computes the gradients of the ``value`` with respect to itself. Returns the result as a
        dictionary. The only key of the dictionary is ``value``. The corresponding gradient is a
        tensor of 1s of identical shape, because :math:`\frac{\partial v}{\partial v} = 1` for any
        :math:`v`.
        """

        return Gradients({value: torch.ones_like(value) for value in self.values})

    @property
    def required_keys(self) -> set[Tensor]:
        return set()

    @property
    def output_keys(self) -> set[Tensor]:
        return self.values
