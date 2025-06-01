import torch
from pytest import mark
from torch import Tensor
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


def test_one_nan():
    aggregator = Sum()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.nan
    result = aggregator(matrix)
    assert result[0].isnan()
    assert_close(result[1:], torch.full_like(result[1:], 10.0))


def test_full_nan():
    aggregator = Sum()
    matrix = torch.full([10, 100], torch.nan)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_inf():
    aggregator = Sum()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.inf
    result = aggregator(matrix)
    assert result[0] == torch.inf
    assert_close(result[1:], torch.full_like(result[1:], 10.0))


def test_full_inf():
    aggregator = Sum()
    matrix = torch.full([10, 100], torch.inf)
    result = aggregator(matrix)
    assert result.eq(torch.full_like(result, torch.inf)).all()


def test_one_neg_inf():
    aggregator = Sum()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = -torch.inf
    result = aggregator(matrix)
    assert result[0] == -torch.inf
    assert_close(result[1:], torch.full_like(result[1:], 10.0))


def test_full_neg_inf():
    aggregator = Sum()
    matrix = torch.full([10, 100], -torch.inf)
    result = aggregator(matrix)
    assert result.eq(torch.full_like(result, -torch.inf)).all()


def test_representations():
    A = Sum()
    assert repr(A) == "Sum()"
    assert str(A) == "Sum"
