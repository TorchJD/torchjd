import pytest
import torch
from unit.conftest import DEVICE

from torchjd.autojac._transform import Gradients, Store

from ._dict_assertions import assert_tensor_dicts_are_close


def test_store():
    """Tests that the Store transform correctly stores gradients in .grad fields."""

    key1 = torch.zeros([], requires_grad=True, device=DEVICE)
    key2 = torch.zeros([1], requires_grad=True, device=DEVICE)
    key3 = torch.zeros([2, 3], requires_grad=True, device=DEVICE)
    value1 = torch.ones([], device=DEVICE)
    value2 = torch.ones([1], device=DEVICE)
    value3 = torch.ones([2, 3], device=DEVICE)
    input = Gradients({key1: value1, key2: value2, key3: value3})

    store = Store([key1, key2, key3])

    output = store(input)
    expected_output = {}

    assert_tensor_dicts_are_close(output, expected_output)

    stored_grads = {key1: key1.grad, key2: key2.grad, key3: key3.grad}
    expected_stored_grads = {key1: value1, key2: value2, key3: value3}

    assert_tensor_dicts_are_close(stored_grads, expected_stored_grads)


def test_store_fails_on_no_requires_grad():
    """
    Tests that the Store transform raises an error when it tries to populate a .grad of a tensor
    that does not require grad.
    """

    key1 = torch.zeros([1], requires_grad=False, device=DEVICE)
    value1 = torch.ones([1], device=DEVICE)
    input = Gradients({key1: value1})

    store = Store([key1])

    with pytest.raises(ValueError):
        store(input)


def test_store_fails_on_no_leaf_and_no_retains_grad():
    """
    Tests that the Store transform raises an error when it tries to populate a .grad of a tensor
    that is not a leaf and that does not retain grad.
    """

    a = torch.tensor([1.0], requires_grad=True, device=DEVICE)
    key1 = 2 * a  # requires_grad=True, but is_leaf=False and retains_grad=False
    value1 = torch.ones([1], device=DEVICE)
    input = Gradients({key1: value1})

    store = Store([key1])

    with pytest.raises(ValueError):
        store(input)
