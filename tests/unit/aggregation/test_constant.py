import pytest
import torch
from torch import Tensor
from unit.aggregation.utils.inputs import (
    matrices,
    scaled_matrices,
    stationary_matrices,
    zero_rank_matrices,
)
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation import ConstantWeighting, WeightedAggregator

# The `ConstantWeighting`'s `weights` parameter must be a vector of length equal to the number of
# rows in the matrix that it will be applied to. Thus, each `ConstantWeighting` is specific to
# matrices of a given number of rows. To test properties on all possible matrices, we have to create
# one `ConstantWeighting` with the right number of weights for each matrix.


def _make_aggregator(matrix: Tensor) -> WeightedAggregator:
    n_rows = matrix.shape[0]
    weights = torch.tensor([1.0 / n_rows] * n_rows)
    weighting = ConstantWeighting(weights)
    aggregator = WeightedAggregator(weighting)
    return aggregator


_matrices_1 = scaled_matrices + zero_rank_matrices
_aggregators_1 = [_make_aggregator(matrix) for matrix in _matrices_1]

_matrices_2 = matrices + stationary_matrices
_aggregators_2 = [_make_aggregator(matrix) for matrix in _matrices_2]


class TestConstant(ExpectedShapeProperty):
    # Override the parametrization of `test_expected_shape_property` to make the test use the right
    # aggregator with each matrix.

    @classmethod
    @pytest.mark.parametrize(["aggregator", "matrix"], zip(_aggregators_1, _matrices_1))
    def test_expected_shape_property(cls, aggregator: WeightedAggregator, matrix: Tensor):
        cls._assert_expected_shape_property(aggregator, matrix)


def test_representations():
    weighting = ConstantWeighting(weights=torch.tensor([1.0, 2.0]))
    assert repr(weighting) == "ConstantWeighting(weights=tensor([1., 2.]))"
    assert str(weighting) == "ConstantWeighting([1., 2.])"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == (
        "WeightedAggregator(weighting=ConstantWeighting(weights=tensor([1., 2.])))"
    )
    assert str(aggregator) == "Constant([1., 2.])"
