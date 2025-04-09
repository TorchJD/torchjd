from collections.abc import Set

import torch
from torch import Tensor

from .base import RequirementError, Transform
from .tensor_dict import EmptyTensorDict, Gradients


class Init(Transform[EmptyTensorDict, Gradients]):
    def __init__(self, values: Set[Tensor]):
        self.values = values

    def __call__(self, input: EmptyTensorDict) -> Gradients:
        r"""
        Computes the gradients of the ``value`` with respect to itself. Returns the result as a
        dictionary. The only key of the dictionary is ``value``. The corresponding gradient is a
        tensor of 1s of identical shape, because :math:`\frac{\partial v}{\partial v} = 1` for any
        :math:`v`.
        """

        return Gradients({value: torch.ones_like(value) for value in self.values})

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        if not input_keys == set():
            raise RequirementError(
                f"The input_keys should be the empty set. Found input_keys {input_keys}."
            )
        return set(self.values)
