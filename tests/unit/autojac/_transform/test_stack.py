from typing import Iterable

import torch
from torch import Tensor

from torchjd.autojac._transform import EmptyTensorDict, Gradients, Stack, Transform

from ._dict_assertions import assert_tensor_dicts_are_close


class FakeGradientsTransform(Transform[EmptyTensorDict, Gradients]):
    """
    Transform that produces gradients filled with ones, for testing purposes. Note that it does the
    same thing as Init, but it does not depend on Init.
    """

    def __init__(self, keys: Iterable[Tensor]):
        self.keys = set(keys)

    def _compute(self, input: EmptyTensorDict) -> Gradients:
        return Gradients({key: torch.ones_like(key) for key in self.keys})

    @property
    def required_keys(self) -> set[Tensor]:
        return set()

    @property
    def output_keys(self) -> set[Tensor]:
        return self.keys


def test_single_key():
    """
    Tests that the Stack transform correctly stacks gradients into a jacobian, in a very simple
    example with 2 transforms sharing the same key.
    """

    key = torch.zeros([3, 4])
    input = EmptyTensorDict()

    transform = FakeGradientsTransform([key])
    stack = Stack([transform, transform])

    output = stack(input)
    expected_output = {key: torch.ones([2, 3, 4])}

    assert_tensor_dicts_are_close(output, expected_output)


def test_disjoint_key_sets():
    """
    Tests that the Stack transform correctly stacks gradients into a jacobian, in an example where
    the output key sets of all of its transforms are disjoint. The missing values should be replaced
    by zeros.
    """

    key1 = torch.zeros([1, 2])
    key2 = torch.zeros([3])
    input = EmptyTensorDict()

    transform1 = FakeGradientsTransform([key1])
    transform2 = FakeGradientsTransform([key2])
    stack = Stack([transform1, transform2])

    output = stack(input)
    expected_output = {
        key1: torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]]),
        key2: torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_overlapping_key_sets():
    """
    Tests that the Stack transform correctly stacks gradients into a jacobian, in an example where
    the output key sets all of its transforms are overlapping (non-empty intersection, but not
    equal). The missing values should be replaced by zeros.
    """

    key1 = torch.zeros([1, 2])
    key2 = torch.zeros([3])
    key3 = torch.zeros([4])
    input = EmptyTensorDict()

    transform12 = FakeGradientsTransform([key1, key2])
    transform23 = FakeGradientsTransform([key2, key3])
    stack = Stack([transform12, transform23])

    output = stack(input)
    expected_output = {
        key1: torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]]),
        key2: torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        key3: torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_empty():
    """Tests that the Stack transform correctly handles an empty list of transforms."""

    stack = Stack([])
    input = EmptyTensorDict({})
    output = stack(input)
    expected_output = EmptyTensorDict({})

    assert_tensor_dicts_are_close(output, expected_output)
