import torch
from pytest import mark
from torch import Tensor
from torch.testing import assert_close

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
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: Mean, matrix: Tensor):
    assert_linear_under_scaling(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: Mean, matrix: Tensor):
    assert_strongly_stationary(aggregator, matrix)


def test_one_nan():
    aggregator = Mean()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.nan
    result = aggregator(matrix)
    assert result[0].isnan()
    assert_close(result[1:], torch.full_like(result[1:], 1.0))


def test_full_nan():
    aggregator = Mean()
    matrix = torch.full([10, 100], torch.nan)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_inf():
    aggregator = Mean()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.inf
    result = aggregator(matrix)
    assert result[0] == torch.inf
    assert_close(result[1:], torch.full_like(result[1:], 1.0))


def test_full_inf():
    aggregator = Mean()
    matrix = torch.full([10, 100], torch.inf)
    result = aggregator(matrix)
    assert result.eq(torch.full_like(result, torch.inf)).all()


def test_one_neg_inf():
    aggregator = Mean()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = -torch.inf
    result = aggregator(matrix)
    assert result[0] == -torch.inf
    assert_close(result[1:], torch.full_like(result[1:], 1.0))


def test_full_neg_inf():
    aggregator = Mean()
    matrix = torch.full([10, 100], -torch.inf)
    result = aggregator(matrix)
    assert result.eq(torch.full_like(result, -torch.inf)).all()


def test_representations():
    A = Mean()
    assert repr(A) == "Mean()"
    assert str(A) == "Mean"
