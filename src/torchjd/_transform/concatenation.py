from typing import Sequence

import torch
from torch import Tensor

from torchjd._transform._utils import _A, _materialize, dicts_union
from torchjd._transform.base import Transform
from torchjd._transform.tensor_dict import Jacobians


class Concatenation(Transform[_A, Jacobians]):
    def __init__(self, transforms: Sequence[Transform[_A, Jacobians]]):
        if len(transforms) == 0:
            raise ValueError("Parameter `transforms` cannot be empty.")

        self.transforms = transforms

        self._required_keys = transforms[0].required_keys
        self._output_keys = {key for transform in transforms for key in transform.output_keys}

        for transform in transforms[1:]:
            if transform.required_keys != self.required_keys:
                raise ValueError("All transforms should require the same set of keys.")

    def _compute(self, input: _A) -> Jacobians:
        results = [transform(input) for transform in self.transforms]
        result = _concatenate(results)
        return result

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._output_keys


def _concatenate(jacobians_dicts: list[Jacobians]) -> Jacobians:
    """
    Transforms a list of tensor dicts into a single dict of (concatenated) tensors. The set of keys
    of the resulting dict is the union of the sets of keys of the input dicts. If a key is absent in
    some input dicts, the corresponding concatenated tensor is filled with zeroes at the positions
    corresponding to those dicts.
    """

    # It is important to first remove duplicate keys before computing their associated
    # concatenated tensor. Otherwise, some computations would be duplicated. Therefore, we first
    # compute unique_keys, and only then, we compute the concatenated tensors.
    unique_keys = dicts_union(jacobians_dicts).keys()
    result = Jacobians({key: _concatenate_one_key(jacobians_dicts, key) for key in unique_keys})
    return result


def _concatenate_one_key(jacobian_dicts: list[Jacobians], input: Tensor) -> Tensor:
    """
    Makes the concatenated tensor corresponding to a given key, from a list of tensor dicts.
    """

    first_dimensions = [jacobian_dict.first_dimension for jacobian_dict in jacobian_dicts]
    optional_jacobians = [jacobian.get(input, None) for jacobian in jacobian_dicts]
    expanded_inputs = [input.expand(dim, *input.shape) for dim in first_dimensions]
    jacobians = _materialize(optional_jacobians, expanded_inputs)
    jacobian = torch.concatenate(jacobians, dim=0)
    return jacobian
