from typing import Iterable

import torch
from torch import Tensor
from unit._transform.utils import assert_tensor_dicts_are_close

from torchjd._transform import Concatenation, EmptyTensorDict, Jacobians, Transform


class FakeJacobiansTransform(Transform[EmptyTensorDict, Jacobians]):
    """
    Transform that produces jacobians filled with ones, of the specified number of rows, for testing
    purposes.
    """

    def __init__(self, keys: Iterable[Tensor], n_rows: int):
        self.keys = set(keys)
        self.n_rows = n_rows

    def _compute(self, input: EmptyTensorDict) -> Jacobians:
        return Jacobians({key: torch.ones(self.n_rows, *key.shape) for key in self.keys})

    @property
    def required_keys(self) -> set[Tensor]:
        return set()

    @property
    def output_keys(self) -> set[Tensor]:
        return self.keys


def test_concatenation_single_key():
    """
    Tests that the Concatenation transform correctly concatenates jacobians, in a very simple
    example with 2 transforms sharing the same key.
    """

    key = torch.zeros([3])
    input = EmptyTensorDict()

    transform1 = FakeJacobiansTransform([key], n_rows=2)
    transform2 = FakeJacobiansTransform([key], n_rows=4)
    concatenation = Concatenation([transform1, transform2])

    output = concatenation(input)
    expected_output = {key: torch.ones(6, 3)}

    assert_tensor_dicts_are_close(output, expected_output)


def test_concatenation_disjoint_key_sets():
    """
    Tests that the Concatenation transform correctly concatenates jacobians, in an example where the
    output key sets of all of its transforms are disjoint. The missing values should be replaced by
    zeros.
    """

    key1 = torch.zeros([2])
    key2 = torch.zeros([])
    input = EmptyTensorDict()

    transform1 = FakeJacobiansTransform([key1], n_rows=2)
    transform2 = FakeJacobiansTransform([key2], n_rows=1)
    concatenation = Concatenation([transform1, transform2])

    output = concatenation(input)
    expected_output = {
        key1: torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
        key2: torch.tensor([0.0, 0.0, 1.0]),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_concatenation_overlapping_key_sets():
    """
    Tests that the Concatenation transform correctly concatenates jacobians, in an example where the
    output key sets all of its transforms are overlapping (non-empty intersection, but not equal).
    The missing values should be replaced by zeros.
    """

    key1 = torch.zeros([2])
    key2 = torch.zeros([])
    key3 = torch.zeros([])
    input = EmptyTensorDict()

    transform12 = FakeJacobiansTransform([key1, key2], n_rows=2)
    transform23 = FakeJacobiansTransform([key2, key3], n_rows=1)
    concatenation = Concatenation([transform12, transform23])

    output = concatenation(input)
    expected_output = {
        key1: torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
        key2: torch.tensor([1.0, 1.0, 1.0]),
        key3: torch.tensor([0.0, 0.0, 1.0]),
    }

    assert_tensor_dicts_are_close(output, expected_output)
