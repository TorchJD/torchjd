import torch
from unit._transform.utils import assert_tensor_dicts_are_close

from torchjd._transform import Gradients, Jacobians, Scaling


def test_single_scaling():
    """Tests that the Scaling transform works correctly when applied to a single input."""

    key = torch.tensor(2.0)
    value = torch.tensor(5.0)
    input = Gradients({key: value})

    scaling = Scaling({key: 0.5})

    output = scaling(input)
    expected_output = {key: 0.5 * value}

    assert_tensor_dicts_are_close(output, expected_output)


def test_multiple_scalings():
    """Tests that the Scaling transform works correctly when applied to several inputs."""

    key1 = torch.tensor(2.0)
    key2 = torch.tensor(3.0)
    key3 = torch.tensor(2.0)
    value = torch.tensor(5.0)
    input = Gradients({key1: value, key2: value, key3: value})

    scaling = Scaling({key1: 0.5, key2: -1.0, key3: 2.0})

    output = scaling(input)
    expected_output = {key1: 0.5 * value, key2: -value, key3: 2 * value}

    assert_tensor_dicts_are_close(output, expected_output)


def test_jacobian_scaling():
    """Tests that the Scaling transform works correctly when applied to a Jacobians."""

    key = torch.tensor([1.0, 2.0])
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    input = Jacobians({key: value})

    scaling = Scaling({key: -0.1})

    output = scaling(input)
    expected_output = {key: -0.1 * value}

    assert_tensor_dicts_are_close(output, expected_output)
