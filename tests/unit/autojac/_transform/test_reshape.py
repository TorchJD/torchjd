import torch
from unit.autojac._transform.utils import assert_tensor_dicts_are_close

from torchjd.autojac._transform import GradientVectors, Reshape


def test_reshape():
    """Tests that the Reshape transform correctly creates gradients from gradient vectors."""

    key1 = torch.zeros([])
    key2 = torch.zeros([1])
    key3 = torch.zeros([2, 3])
    value1 = torch.tensor([1.0])
    value2 = torch.tensor([2.0])
    value3 = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    input = GradientVectors({key1: value1, key2: value2, key3: value3})

    reshape = Reshape([key1, key2, key3])

    output = reshape(input)
    expected_output = {
        key1: torch.tensor(1.0),
        key2: torch.tensor([2.0]),
        key3: torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
    }

    assert_tensor_dicts_are_close(output, expected_output)
