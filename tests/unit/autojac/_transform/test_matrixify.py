import torch

from torchjd.autojac._transform import Jacobians, Matrixify

from .utils import assert_tensor_dicts_are_close


def test_matrixify():
    """Tests that the Matrixify transform correctly creates matrices from the jacobians."""

    n_outputs = 5
    key1 = torch.zeros([])
    key2 = torch.zeros([1])
    key3 = torch.zeros([2, 3])
    value1 = torch.tensor([1.0] * n_outputs)
    value2 = torch.tensor([[2.0]] * n_outputs)
    value3 = torch.tensor([[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]] * n_outputs)
    input = Jacobians({key1: value1, key2: value2, key3: value3})

    matrixify = Matrixify([key1, key2, key3])

    output = matrixify(input)
    expected_output = {
        key1: torch.tensor([[1.0]] * n_outputs),
        key2: torch.tensor([[2.0]] * n_outputs),
        key3: torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]] * n_outputs),
    }

    assert_tensor_dicts_are_close(output, expected_output)
