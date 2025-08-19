from pytest import mark
from torch import Tensor, tensor
from torch.testing import assert_close

from torchjd.aggregation import Sum

from ._asserts import (
    assert_expected_structure,
    assert_linear_under_scaling,
    assert_permutation_invariant,
    assert_strongly_stationary,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(Sum(), matrix) for matrix in scaled_matrices]
typical_pairs = [(Sum(), matrix) for matrix in typical_matrices]
non_strong_pairs = [(Sum(), matrix) for matrix in non_strong_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: Sum, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: Sum, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: Sum, matrix: Tensor):
    assert_linear_under_scaling(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: Sum, matrix: Tensor):
    assert_strongly_stationary(aggregator, matrix)


def test_value():
    """Test that the output values are fixed (on cpu)."""

    A = Sum()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    assert_close(A(J), tensor([2.0, 2.0, 2.0]), rtol=0, atol=1e-4)


def test_representations():
    A = Sum()
    assert repr(A) == "Sum()"
    assert str(A) == "Sum"
