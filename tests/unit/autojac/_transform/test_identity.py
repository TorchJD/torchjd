import torch

from torchjd.autojac._transform import Gradients, Identity

from .utils import assert_tensor_dicts_are_close


def test_identity():
    """Tests that the Identity transform makes no change to its input."""

    key1 = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    key2 = torch.tensor([1.0, 3.0, 5.0])
    value1 = torch.ones_like(key1)
    value2 = torch.ones_like(key2)
    input = Gradients({key1: value1, key2: value2})

    identity = Identity([key1, key2])

    output = identity(input)
    expected_output = input

    assert_tensor_dicts_are_close(output, expected_output)
