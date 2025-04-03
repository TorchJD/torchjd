import torch
from pytest import raises

from torchjd.autojac._transform import Diagonalize, Gradients, RequirementError
from torchjd.autojac._transform.ordered_set import OrderedSet

from ._dict_assertions import assert_tensor_dicts_are_close


def test_single_input():
    """Tests that the Diagonalize transform works when given a single input."""

    key = torch.tensor([1.0, 2.0, 3.0])
    value = torch.ones_like(key)
    input = Gradients({key: value})

    diag = Diagonalize(OrderedSet([key]))

    output = diag(input)
    expected_output = {key: torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])}

    assert_tensor_dicts_are_close(output, expected_output)


def test_multiple_inputs():
    """Tests that the Diagonalize transform works when given multiple inputs."""

    key1 = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    key2 = torch.tensor([1.0, 3.0, 5.0])
    key3 = torch.tensor(1.0)
    value1 = torch.ones_like(key1)
    value2 = torch.ones_like(key2)
    value3 = torch.ones_like(key3)
    input = Gradients({key1: value1, key2: value2, key3: value3})

    diag = Diagonalize(OrderedSet([key1, key2, key3]))

    output = diag(input)
    expected_output = {
        key1: torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ),
        key2: torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
        ),
        key3: torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_permute_order():
    """
    Tests that the Diagonalize transform outputs a permuted mapping when its keys are permuted.
    """

    key1 = torch.tensor(2.0)
    key2 = torch.tensor(1.0)
    value1 = torch.ones_like(key1)
    value2 = torch.ones_like(key2)
    input = Gradients({key1: value1, key2: value2})

    permuted_diag = Diagonalize(OrderedSet([key2, key1]))
    diag = Diagonalize(OrderedSet([key1, key2]))

    permuted_output = permuted_diag(input)
    output = {key1: permuted_output[key2], key2: permuted_output[key1]}  # un-permute
    expected_output = diag(input)

    assert_tensor_dicts_are_close(output, expected_output)


def test_check_keys():
    """
    Tests that the `check_keys` method works correctly. The input_keys must match the stored
    considered keys.
    """

    key1 = torch.tensor([1.0])
    key2 = torch.tensor([1.0])
    diag = Diagonalize(OrderedSet([key1]))

    output_keys = diag.check_keys({key1})
    assert output_keys == {key1}

    with raises(RequirementError):
        diag.check_keys(set())

    with raises(RequirementError):
        diag.check_keys({key1, key2})
