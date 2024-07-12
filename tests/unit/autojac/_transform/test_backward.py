import torch

from torchjd.autojac._transform import Backward, Gradients

from ._dict_assertions import assert_tensor_dicts_are_close


def test_backward():
    """Tests that the `Backward` transform correctly backwards gradients in .grad fields."""

    key1 = torch.zeros([], requires_grad=True)
    key2 = torch.zeros([1], requires_grad=True)
    key3 = torch.zeros([2, 3], requires_grad=True)
    value1 = torch.ones([])
    value2 = torch.ones([1])
    value3 = torch.ones([2, 3])
    input = Gradients({key1: value1, key2: value2, key3: value3})

    backward = Backward([key1, key2, key3])

    output = backward(input)
    expected_output = {}

    assert_tensor_dicts_are_close(output, expected_output)

    stored_grads = {key1: key1.grad, key2: key2.grad, key3: key3.grad}
    expected_stored_grads = {key1: value1, key2: value2, key3: value3}

    assert_tensor_dicts_are_close(stored_grads, expected_stored_grads)
