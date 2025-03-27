from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch import Tensor
from unit._utils import ExceptionContext

from torchjd.aggregation import Constant

from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices
from ._property_testers import (
    ExpectedStructureProperty,
    LinearUnderScalingProperty,
    StrongStationarityProperty,
)

# The weights must be a vector of length equal to the number of rows in the matrix that it will be
# applied to. Thus, each `Constant` instance is specific to matrices of a given number of rows. To
# test properties on all possible matrices, we have to create one `Constant` with the right number
# of weights for each matrix.


def _make_aggregator(matrix: Tensor) -> Constant:
    n_rows = matrix.shape[0]
    weights = torch.tensor([1.0 / n_rows] * n_rows)
    return Constant(weights)


_matrices_1 = scaled_matrices + typical_matrices
_aggregators_1 = [_make_aggregator(matrix) for matrix in _matrices_1]

_matrices_2 = typical_matrices
_aggregators_2 = [_make_aggregator(matrix) for matrix in _matrices_2]

_matrices_3 = non_strong_matrices
_aggregators_3 = [_make_aggregator(matrix) for matrix in _matrices_3]


class TestConstant(
    ExpectedStructureProperty, LinearUnderScalingProperty, StrongStationarityProperty
):
    # Override the parametrization of `test_expected_structure_property` to make the test use the
    # right aggregator with each matrix.

    @classmethod
    @mark.parametrize(["aggregator", "matrix"], zip(_aggregators_1, _matrices_1))
    def test_expected_structure_property(cls, aggregator: Constant, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)

    @classmethod
    @mark.parametrize(["aggregator", "matrix"], zip(_aggregators_2, _matrices_2))
    def test_linear_under_scaling_property(cls, aggregator: Constant, matrix: Tensor):
        cls._assert_linear_under_scaling_property(aggregator, matrix)

    @classmethod
    @mark.parametrize(["aggregator", "matrix"], zip(_aggregators_3, _matrices_3))
    def test_stationarity_property(cls, aggregator: Constant, matrix: Tensor):
        cls._assert_stationarity_property(aggregator, matrix)


@mark.parametrize(
    ["weights_shape", "expectation"],
    [
        ([], raises(ValueError)),
        ([0], does_not_raise()),
        ([1], does_not_raise()),
        ([10], does_not_raise()),
        ([0, 0], raises(ValueError)),
        ([0, 1], raises(ValueError)),
        ([1, 1], raises(ValueError)),
        ([1, 1, 1], raises(ValueError)),
        ([1, 1, 1, 1], raises(ValueError)),
        ([1, 1, 1, 1, 1], raises(ValueError)),
    ],
)
def test_weights_shape_check(weights_shape: list[int], expectation: ExceptionContext):
    weights = torch.ones(weights_shape)
    with expectation:
        _ = Constant(weights=weights)


@mark.parametrize(
    ["weights_shape", "n_rows", "expectation"],
    [
        ([0], 0, does_not_raise()),
        ([1], 1, does_not_raise()),
        ([5], 5, does_not_raise()),
        ([0], 1, raises(ValueError)),
        ([1], 0, raises(ValueError)),
        ([4], 5, raises(ValueError)),
        ([5], 4, raises(ValueError)),
    ],
)
def test_matrix_shape_check(weights_shape: list[int], n_rows: int, expectation: ExceptionContext):
    matrix = torch.ones([n_rows, 5])
    weights = torch.ones(weights_shape)
    aggregator = Constant(weights)

    with expectation:
        _ = aggregator(matrix)


def test_representations():
    A = Constant(weights=torch.tensor([1.0, 2.0], device="cpu"))
    assert repr(A) == "Constant(weights=tensor([1., 2.]))"
    assert str(A) == "Constant([1., 2.])"
