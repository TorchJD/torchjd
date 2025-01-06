import torch
from pytest import mark
from torch import Tensor

from torchjd.aggregation import Constant

from ._inputs import matrices, scaled_matrices, stationary_matrices, zero_rank_matrices
from ._property_testers import ExpectedStructureProperty

# The weights must be a vector of length equal to the number of rows in the matrix that it will be
# applied to. Thus, each `Constant` instance is specific to matrices of a given number of rows. To
# test properties on all possible matrices, we have to create one `Constant` with the right number
# of weights for each matrix.


def _make_aggregator(matrix: Tensor) -> Constant:
    n_rows = matrix.shape[0]
    weights = torch.tensor([1.0 / n_rows] * n_rows)
    return Constant(weights)


_matrices_1 = scaled_matrices + zero_rank_matrices
_aggregators_1 = [_make_aggregator(matrix) for matrix in _matrices_1]

_matrices_2 = matrices + stationary_matrices
_aggregators_2 = [_make_aggregator(matrix) for matrix in _matrices_2]


class TestConstant(ExpectedStructureProperty):
    # Override the parametrization of `test_expected_structure_property` to make the test use the
    # right aggregator with each matrix.

    @classmethod
    @mark.parametrize(["aggregator", "matrix"], zip(_aggregators_1, _matrices_1))
    def test_expected_structure_property(cls, aggregator: Constant, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)


def test_representations():
    A = Constant(weights=torch.tensor([1.0, 2.0], device="cpu"))
    assert repr(A) == "Constant(weights=tensor([1., 2.]))"
    assert str(A) == "Constant([1., 2.])"
