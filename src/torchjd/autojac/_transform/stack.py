from typing import Sequence

import torch
from torch import Tensor

from ._utils import _A, _materialize, dicts_union
from .base import Transform
from .tensor_dict import Gradients, Jacobians


class Stack(Transform[_A, Jacobians]):
    def __init__(self, transforms: Sequence[Transform[_A, Gradients]]):
        self.transforms = transforms

        self._required_keys = {key for transform in transforms for key in transform.required_keys}
        self._output_keys = {key for transform in transforms for key in transform.output_keys}

        for transform in transforms:
            if transform.required_keys != self.required_keys:
                raise ValueError("All transforms should require the same set of keys.")

    def _compute(self, input: _A) -> Jacobians:
        results = [transform(input) for transform in self.transforms]
        result = _stack(results)
        return result

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._output_keys


def _stack(gradient_dicts: list[Gradients]) -> Jacobians:
    """
    Transforms a list of tensor dicts into a single dict of (stacked) tensors. The set of keys of
    the resulting dict is the union of the sets of keys of the input dicts.
    If a key is absent in some input dicts, the corresponding stacked tensor is filled with zeroes
    at the positions corresponding to those dicts.
    """

    # It is important to first remove duplicate keys before computing their associated
    # stacked tensor. Otherwise, some computations would be duplicated. Therefore, we first compute
    # unique_keys, and only then, we compute the stacked tensors.
    unique_keys = dicts_union(gradient_dicts).keys()
    result = Jacobians({key: _stack_one_key(gradient_dicts, key) for key in unique_keys})
    return result


def _stack_one_key(gradient_dicts: list[Gradients], input: Tensor) -> Tensor:
    """
    Makes the stacked tensor corresponding to a given key, from a list of tensor dicts.
    """

    optional_gradients = [gradients.get(input, None) for gradients in gradient_dicts]
    gradients = _materialize(optional_gradients, [input] * len(optional_gradients))
    jacobian = torch.stack(gradients, dim=0)
    return jacobian
