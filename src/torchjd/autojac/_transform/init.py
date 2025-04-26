from collections.abc import Set

import torch
from torch import Tensor

from .base import RequirementError, Transform
from .tensor_dict import EmptyTensorDict, Gradients


class Init(Transform[EmptyTensorDict, Gradients]):
    """
    Transform returning Gradients filled with ones, corresponding to the gradients of the provided
    values with respect to themselves.

    :param values: Tensors for which Gradients must be returned.
    """

    def __init__(self, values: Set[Tensor]):
        self.values = values

    def __call__(self, input: EmptyTensorDict) -> Gradients:
        return Gradients({value: torch.ones_like(value) for value in self.values})

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        if not input_keys == set():
            raise RequirementError(
                f"The input_keys should be the empty set. Found input_keys {input_keys}."
            )
        return set(self.values)
