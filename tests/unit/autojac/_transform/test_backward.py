import pytest
import torch

from torchjd.autojac._transform import Backward, Gradients

from ._dict_assertions import assert_tensor_dicts_are_close


def test_backward_leaves():
    """Tests that the `Backward` transform correctly stores gradients in ``.grad`` fields for leaves
    tensors."""

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


@pytest.mark.parametrize("shape", [(3, 5), (7, 8), (1, 101)])
def test_backward_non_leaves(shape: tuple[int]):
    """Tests that the `Backward` transform correctly backpropagates gradients for non-leaves
    tensors."""

    key1 = torch.zeros([shape[1]], requires_grad=True)
    key2 = torch.zeros([shape[0]], requires_grad=True)

    matrix = torch.randn(shape)
    non_leaf = matrix @ key1 + key2

    input = Gradients({non_leaf: torch.ones_like(non_leaf)})

    backward = Backward([non_leaf])

    output = backward(input)
    expected_output = {}

    assert_tensor_dicts_are_close(output, expected_output)

    stored_grads = {key1: key1.grad, key2: key2.grad}
    expected_stored_grads = {key1: matrix.sum(dim=0), key2: torch.ones_like(key2)}

    assert_tensor_dicts_are_close(stored_grads, expected_stored_grads)
