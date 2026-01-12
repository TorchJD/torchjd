from pytest import mark, raises
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.tensors import ones_, tensor_, zeros_

from torchjd.autojac._transform import AccumulateGrad, AccumulateJac


def test_single_grad_accumulation():
    """
    Tests that the AccumulateGrad transform correctly accumulates gradients in .grad fields when run
    once.
    """

    key1 = zeros_([], requires_grad=True)
    key2 = zeros_([1], requires_grad=True)
    key3 = zeros_([2, 3], requires_grad=True)
    value1 = ones_([])
    value2 = ones_([1])
    value3 = ones_([2, 3])
    input = {key1: value1, key2: value2, key3: value3}

    accumulate = AccumulateGrad()

    output = accumulate(input)
    expected_output = {}

    assert_tensor_dicts_are_close(output, expected_output)

    grads = {key1: key1.grad, key2: key2.grad, key3: key3.grad}
    expected_grads = {key1: value1, key2: value2, key3: value3}

    assert_tensor_dicts_are_close(grads, expected_grads)


@mark.parametrize("iterations", [1, 2, 4, 10, 13])
def test_multiple_grad_accumulations(iterations: int):
    """
    Tests that the AccumulateGrad transform correctly accumulates gradients in .grad fields when run
    `iterations` times.
    """

    key1 = zeros_([], requires_grad=True)
    key2 = zeros_([1], requires_grad=True)
    key3 = zeros_([2, 3], requires_grad=True)
    value1 = ones_([])
    value2 = ones_([1])
    value3 = ones_([2, 3])

    accumulate = AccumulateGrad()

    for i in range(iterations):
        # Clone values to ensure that we accumulate values that are not ever used afterwards
        input = {key1: value1.clone(), key2: value2.clone(), key3: value3.clone()}
        accumulate(input)

    grads = {key1: key1.grad, key2: key2.grad, key3: key3.grad}
    expected_grads = {
        key1: iterations * value1,
        key2: iterations * value2,
        key3: iterations * value3,
    }

    assert_tensor_dicts_are_close(grads, expected_grads)


def test_accumulate_grad_fails_when_no_requires_grad():
    """
    Tests that the AccumulateGrad transform raises an error when it tries to populate a .grad of a
    tensor that does not require grad.
    """

    key = zeros_([1], requires_grad=False)
    value = ones_([1])
    input = {key: value}

    accumulate = AccumulateGrad()

    with raises(ValueError):
        accumulate(input)


def test_accumulate_grad_fails_when_no_leaf_and_no_retains_grad():
    """
    Tests that the AccumulateGrad transform raises an error when it tries to populate a .grad of a
    tensor that is not a leaf and that does not retain grad.
    """

    key = tensor_([1.0], requires_grad=True) * 2
    value = ones_([1])
    input = {key: value}

    accumulate = AccumulateGrad()

    with raises(ValueError):
        accumulate(input)


def test_accumulate_grad_check_keys():
    """Tests that the `check_keys` method works correctly for AccumulateGrad."""

    key = tensor_([1.0], requires_grad=True)
    accumulate = AccumulateGrad()

    output_keys = accumulate.check_keys({key})
    assert output_keys == set()


def test_single_jac_accumulation():
    """
    Tests that the AccumulateJac transform correctly accumulates jacobians in .jac fields when run
    once.
    """

    key1 = zeros_([], requires_grad=True)
    key2 = zeros_([1], requires_grad=True)
    key3 = zeros_([2, 3], requires_grad=True)
    value1 = ones_([4])
    value2 = ones_([4, 1])
    value3 = ones_([4, 2, 3])
    input = {key1: value1, key2: value2, key3: value3}

    accumulate = AccumulateJac()

    output = accumulate(input)
    expected_output = {}

    assert_tensor_dicts_are_close(output, expected_output)

    jacs = {key1: key1.jac, key2: key2.jac, key3: key3.jac}
    expected_jacs = {key1: value1, key2: value2, key3: value3}

    assert_tensor_dicts_are_close(jacs, expected_jacs)


@mark.parametrize("iterations", [1, 2, 4, 10, 13])
def test_multiple_jac_accumulations(iterations: int):
    """
    Tests that the AccumulateJac transform correctly accumulates jacobians in .jac fields when run
    `iterations` times.
    """

    key1 = zeros_([], requires_grad=True)
    key2 = zeros_([1], requires_grad=True)
    key3 = zeros_([2, 3], requires_grad=True)
    value1 = ones_([4])
    value2 = ones_([4, 1])
    value3 = ones_([4, 2, 3])

    accumulate = AccumulateJac()

    for i in range(iterations):
        # Clone values to ensure that we accumulate values that are not ever used afterwards
        input = {key1: value1.clone(), key2: value2.clone(), key3: value3.clone()}
        accumulate(input)

    jacs = {key1: key1.jac, key2: key2.jac, key3: key3.jac}
    expected_jacs = {
        key1: iterations * value1,
        key2: iterations * value2,
        key3: iterations * value3,
    }

    assert_tensor_dicts_are_close(jacs, expected_jacs)


def test_accumulate_jac_fails_when_no_requires_grad():
    """
    Tests that the AccumulateJac transform raises an error when it tries to populate a .jac of a
    tensor that does not require grad.
    """

    key = zeros_([1], requires_grad=False)
    value = ones_([4, 1])
    input = {key: value}

    accumulate = AccumulateJac()

    with raises(ValueError):
        accumulate(input)


def test_accumulate_jac_fails_when_no_leaf_and_no_retains_grad():
    """
    Tests that the AccumulateJac transform raises an error when it tries to populate a .jac of a
    tensor that is not a leaf and that does not retain grad.
    """

    key = tensor_([1.0], requires_grad=True) * 2
    value = ones_([4, 1])
    input = {key: value}

    accumulate = AccumulateJac()

    with raises(ValueError):
        accumulate(input)


def test_accumulate_jac_fails_when_shape_mismatch():
    """
    Tests that the AccumulateJac transform raises an error when the jacobian shape does not match
    the parameter shape (ignoring the first dimension).
    """

    key = zeros_([2, 3], requires_grad=True)
    value = ones_([4, 3, 2])  # Wrong shape: should be [4, 2, 3], not [4, 3, 2]
    input = {key: value}

    accumulate = AccumulateJac()

    with raises(RuntimeError):
        accumulate(input)


def test_accumulate_jac_check_keys():
    """Tests that the `check_keys` method works correctly for AccumulateJac."""

    key = tensor_([1.0], requires_grad=True)
    accumulate = AccumulateJac()

    output_keys = accumulate.check_keys({key})
    assert output_keys == set()
