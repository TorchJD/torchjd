import torch

from torchjd.autojac._transform import Diagonalize, Gradients

from ._dict_assertions import assert_tensor_dicts_are_close


def test_single_input():
    """Tests that the Diagonalize transform works when given a single input."""

    key = torch.tensor([1.0, 2.0, 3.0])
    value = torch.ones_like(key)
    input = Gradients({key: value})

    diag = Diagonalize([key])

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

    diag = Diagonalize([key1, key2, key3])

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

    permuted_diag = Diagonalize([key2, key1])
    diag = Diagonalize([key1, key2])

    permuted_output = permuted_diag(input)
    output = {key1: permuted_output[key2], key2: permuted_output[key1]}  # un-permute
    expected_output = diag(input)

    assert_tensor_dicts_are_close(output, expected_output)
