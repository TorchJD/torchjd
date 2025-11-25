import math

import torch
from pytest import mark, raises

from tests.device import DEVICE
from tests.utils.dict_assertions import assert_tensor_dicts_are_close
from tests.utils.tensors import rand_, tensor_, zeros_
from torchjd.aggregation import Random
from torchjd.autojac._transform import OrderedSet, RequirementError
from torchjd.autojac._transform._aggregate import _AggregateMatrices, _Matrixify, _Reshape
from torchjd.autojac._transform._base import TensorDict


def _make_jacobian_matrices(n_outputs: int, rng: torch.Generator) -> TensorDict:
    jacobian_shapes = [[n_outputs, math.prod(shape)] for shape in _param_shapes]
    jacobian_list = [rand_(shape, generator=rng) for shape in jacobian_shapes]
    jacobian_matrices = {key: jac for key, jac in zip(_keys, jacobian_list)}
    return jacobian_matrices


_param_shapes = [
    [],
    [1],
    [2],
    [5],
    [1, 1],
    [2, 3],
    [5, 5],
    [1, 1, 1],
    [2, 3, 4],
    [5, 5, 5],
    [1, 1, 1, 1],
    [2, 3, 4, 5],
    [5, 5, 5, 5],
]
_keys = [zeros_(shape) for shape in _param_shapes]

_rng = torch.Generator(device=DEVICE)
_rng.manual_seed(0)
_jacobian_matrix_dicts = [_make_jacobian_matrices(n_outputs, _rng) for n_outputs in [1, 2, 5]]


@mark.parametrize("jacobian_matrices", _jacobian_matrix_dicts)
def test_aggregate_matrices_output_structure(jacobian_matrices: TensorDict):
    """
    Tests that applying _AggregateMatrices to various dictionaries of jacobian matrices gives an
    output of the desired structure.
    """

    aggregate_matrices = _AggregateMatrices(Random(), key_order=OrderedSet(_keys))
    gradient_vectors = aggregate_matrices(jacobian_matrices)

    assert set(jacobian_matrices.keys()) == set(gradient_vectors.keys())

    for key in jacobian_matrices.keys():
        assert gradient_vectors[key].numel() == jacobian_matrices[key][0].numel()


def test_aggregate_matrices_empty_dict():
    """Tests that applying _AggregateMatrices to an empty input gives an empty output."""

    aggregate_matrices = _AggregateMatrices(Random(), key_order=OrderedSet([]))
    gradient_vectors = aggregate_matrices({})
    assert len(gradient_vectors) == 0


def test_matrixify():
    """Tests that the Matrixify transform correctly creates matrices from the jacobians."""

    n_outputs = 5
    key1 = zeros_([])
    key2 = zeros_([1])
    key3 = zeros_([2, 3])
    value1 = tensor_([1.0] * n_outputs)
    value2 = tensor_([[2.0]] * n_outputs)
    value3 = tensor_([[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]] * n_outputs)
    input = {key1: value1, key2: value2, key3: value3}

    matrixify = _Matrixify()

    output = matrixify(input)
    expected_output = {
        key1: tensor_([[1.0]] * n_outputs),
        key2: tensor_([[2.0]] * n_outputs),
        key3: tensor_([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]] * n_outputs),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_reshape():
    """Tests that the Reshape transform correctly creates gradients from gradient vectors."""

    key1 = zeros_([])
    key2 = zeros_([1])
    key3 = zeros_([2, 3])
    value1 = tensor_([1.0])
    value2 = tensor_([2.0])
    value3 = tensor_([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    input = {key1: value1, key2: value2, key3: value3}

    reshape = _Reshape()

    output = reshape(input)
    expected_output = {
        key1: tensor_(1.0),
        key2: tensor_([2.0]),
        key3: tensor_([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_aggregate_matrices_check_keys():
    """
    Tests that the `check_keys` method works correctly: the input_keys must match the stored
    key_order.
    """

    key1 = tensor_([1.0])
    key2 = tensor_([2.0])
    key3 = tensor_([2.0])
    aggregate = _AggregateMatrices(Random(), OrderedSet([key2, key1]))

    output_keys = aggregate.check_keys({key1, key2})
    assert output_keys == {key1, key2}

    with raises(RequirementError):
        aggregate.check_keys({key1})

    with raises(RequirementError):
        aggregate.check_keys({key1, key2, key3})


def test_matrixify_check_keys():
    """Tests that the `check_keys` method works correctly."""

    key1 = tensor_([1.0])
    key2 = tensor_([2.0])
    matrixify = _Matrixify()

    output_keys = matrixify.check_keys({key1, key2})
    assert output_keys == {key1, key2}


def test_reshape_check_keys():
    """Tests that the `check_keys` method works correctly."""

    key1 = tensor_([1.0])
    key2 = tensor_([2.0])
    reshape = _Reshape()

    output_keys = reshape.check_keys({key1, key2})
    assert output_keys == {key1, key2}
