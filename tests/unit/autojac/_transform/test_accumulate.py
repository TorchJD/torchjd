import torch
from pytest import mark, raises
from unit.conftest import DEVICE

from torchjd.autojac._transform import Accumulate, Gradients

from ._dict_assertions import assert_tensor_dicts_are_close


def test_single_accumulation():
    """
    Tests that the Accumulate transform correctly accumulates gradients in .grad fields when run
    once.
    """

    key1 = torch.zeros([], requires_grad=True, device=DEVICE)
    key2 = torch.zeros([1], requires_grad=True, device=DEVICE)
    key3 = torch.zeros([2, 3], requires_grad=True, device=DEVICE)
    value1 = torch.ones([], device=DEVICE)
    value2 = torch.ones([1], device=DEVICE)
    value3 = torch.ones([2, 3], device=DEVICE)
    input = Gradients({key1: value1, key2: value2, key3: value3})

    accumulate = Accumulate([key1, key2, key3])

    output = accumulate(input)
    expected_output = {}

    assert_tensor_dicts_are_close(output, expected_output)

    grads = {key1: key1.grad, key2: key2.grad, key3: key3.grad}
    expected_grads = {key1: value1, key2: value2, key3: value3}

    assert_tensor_dicts_are_close(grads, expected_grads)


@mark.parametrize("iterations", [1, 2, 4, 10, 13])
def test_multiple_accumulation(iterations: int):
    """
    Tests that the Accumulate transform correctly accumulates gradients in .grad fields when run
    `iterations` times.
    """

    key1 = torch.zeros([], requires_grad=True, device=DEVICE)
    key2 = torch.zeros([1], requires_grad=True, device=DEVICE)
    key3 = torch.zeros([2, 3], requires_grad=True, device=DEVICE)
    value1 = torch.ones([], device=DEVICE)
    value2 = torch.ones([1], device=DEVICE)
    value3 = torch.ones([2, 3], device=DEVICE)
    input = Gradients({key1: value1, key2: value2, key3: value3})

    accumulate = Accumulate([key1, key2, key3])

    for i in range(iterations):
        accumulate(input)

    grads = {key1: key1.grad, key2: key2.grad, key3: key3.grad}
    expected_grads = {
        key1: iterations * value1,
        key2: iterations * value2,
        key3: iterations * value3,
    }

    assert_tensor_dicts_are_close(grads, expected_grads)


def test_accumulate_fails_on_no_requires_grad():
    """
    Tests that the Accumulate transform raises an error when it tries to populate a .grad of a
    tensor that does not require grad.
    """

    key1 = torch.zeros([1], requires_grad=False, device=DEVICE)
    value1 = torch.ones([1], device=DEVICE)
    input = Gradients({key1: value1})

    accumulate = Accumulate([key1])

    with raises(ValueError):
        accumulate(input)


def test_accumulate_fails_on_no_leaf_and_no_retains_grad():
    """
    Tests that the Accumulate transform raises an error when it tries to populate a .grad of a
    tensor that is not a leaf and that does not retain grad.
    """

    a = torch.tensor([1.0], requires_grad=True, device=DEVICE)
    key1 = 2 * a  # requires_grad=True, but is_leaf=False and retains_grad=False
    value1 = torch.ones([1], device=DEVICE)
    input = Gradients({key1: value1})

    accumulate = Accumulate([key1])

    with raises(ValueError):
        accumulate(input)
