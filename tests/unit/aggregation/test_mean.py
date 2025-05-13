from pytest import mark
from torch import Tensor

from torchjd.aggregation import Mean

from ._asserts import (
    assert_expected_structure,
    assert_linear_under_scaling,
    assert_permutation_invariant,
    assert_strongly_stationary,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(Mean(), matrix) for matrix in scaled_matrices]
typical_pairs = [(Mean(), matrix) for matrix in typical_matrices]
non_strong_pairs = [(Mean(), matrix) for matrix in non_strong_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: Mean, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: Mean, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix, n_runs=5, atol=5e-04, rtol=1e-05)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: Mean, matrix: Tensor):
    assert_linear_under_scaling(aggregator, matrix, n_runs=5, atol=1e-02, rtol=0)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: Mean, matrix: Tensor):
    assert_strongly_stationary(aggregator, matrix, threshold=1e-03)


def test_representations():
    A = Mean()
    assert repr(A) == "Mean()"
    assert str(A) == "Mean"
