import math
from collections import OrderedDict

import torch
from pytest import mark, raises
from torch import Tensor
from unit.conftest import DEVICE

from torchjd.aggregation import Random
from torchjd.autojac._transform import GradientVectors, JacobianMatrices, Jacobians
from torchjd.autojac._transform.aggregate import _AggregateMatrices, _Matrixify, _Reshape

from ._dict_assertions import assert_tensor_dicts_are_close


def _make_jacobian_matrices(n_outputs: int, rng: torch.Generator) -> JacobianMatrices:
    jacobian_shapes = [[n_outputs, math.prod(shape)] for shape in _param_shapes]
    jacobian_list = [torch.rand(shape, generator=rng) for shape in jacobian_shapes]
    jacobian_matrices = JacobianMatrices({key: jac for key, jac in zip(_keys, jacobian_list)})
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
_keys = [torch.zeros(shape) for shape in _param_shapes]

_rng = torch.Generator(device=DEVICE)
_rng.manual_seed(0)
_jacobian_matrix_dicts = [_make_jacobian_matrices(n_outputs, _rng) for n_outputs in [1, 2, 5]]


@mark.parametrize("jacobian_matrices", _jacobian_matrix_dicts)
def test_aggregate_matrices_output_structure(jacobian_matrices: JacobianMatrices):
    """
    Tests that applying _AggregateMatrices to various dictionaries of jacobian matrices gives an
    output of the desired structure.
    """

    aggregate_matrices = _AggregateMatrices(Random(), key_order=_keys)
    gradient_vectors = aggregate_matrices(jacobian_matrices)

    assert set(jacobian_matrices.keys()) == set(gradient_vectors.keys())

    for key in jacobian_matrices.keys():
        assert gradient_vectors[key].numel() == jacobian_matrices[key][0].numel()


def test_aggregate_matrices_empty_dict():
    """Tests that applying _AggregateMatrices to an empty input gives an empty output."""

    aggregate_matrices = _AggregateMatrices(Random(), key_order=[])
    gradient_vectors = aggregate_matrices(JacobianMatrices({}))
    assert len(gradient_vectors) == 0


@mark.parametrize(
    ["united_gradient_vector", "jacobian_matrices"],
    [
        (
            torch.ones(10),
            {  # Total number of parameters according to the united gradient vector: 10
                torch.ones(5): torch.ones(2, 5),
                torch.ones(4): torch.ones(2, 4),
            },
        ),  # Total number of parameters according to the jacobian matrices: 9
        (
            torch.ones(10),
            {  # Total number of parameters according to the united gradient vector: 10
                torch.ones(5): torch.ones(2, 5),
                torch.ones(3): torch.ones(2, 3),
                torch.ones(3): torch.ones(2, 3),
            },
        ),  # Total number of parameters according to the jacobian matrices: 11
    ],
)
def test_disunite_wrong_vector_length(
    united_gradient_vector: Tensor, jacobian_matrices: dict[Tensor, Tensor]
):
    """
    Tests that the _disunite method raises a ValueError when used on vectors of the wrong length.
    """

    with raises(ValueError):
        _AggregateMatrices._disunite(united_gradient_vector, OrderedDict(jacobian_matrices))


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

    matrixify = _Matrixify([key1, key2, key3])

    output = matrixify(input)
    expected_output = {
        key1: torch.tensor([[1.0]] * n_outputs),
        key2: torch.tensor([[2.0]] * n_outputs),
        key3: torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]] * n_outputs),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_reshape():
    """Tests that the Reshape transform correctly creates gradients from gradient vectors."""

    key1 = torch.zeros([])
    key2 = torch.zeros([1])
    key3 = torch.zeros([2, 3])
    value1 = torch.tensor([1.0])
    value2 = torch.tensor([2.0])
    value3 = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    input = GradientVectors({key1: value1, key2: value2, key3: value3})

    reshape = _Reshape([key1, key2, key3])

    output = reshape(input)
    expected_output = {
        key1: torch.tensor(1.0),
        key2: torch.tensor([2.0]),
        key3: torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
    }

    assert_tensor_dicts_are_close(output, expected_output)
