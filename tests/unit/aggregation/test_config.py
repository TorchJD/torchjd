import torch
from pytest import mark
from torch import Tensor

from torchjd.aggregation import ConFIG

from ._asserts import (
    assert_expected_structure,
    assert_linear_under_scaling,
    assert_non_differentiable,
    assert_permutation_invariant,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(ConFIG(), matrix) for matrix in scaled_matrices]
typical_pairs = [(ConFIG(), matrix) for matrix in typical_matrices]
non_strong_pairs = [(ConFIG(), matrix) for matrix in non_strong_matrices]
requires_grad_pairs = [(ConFIG(), torch.ones(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: ConFIG, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: ConFIG, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: ConFIG, matrix: Tensor):
    assert_linear_under_scaling(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: ConFIG, matrix: Tensor):
    assert_non_differentiable(aggregator, matrix)


def test_one_nan():
    aggregator = ConFIG()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.nan
    result = aggregator(matrix)
    assert result.isnan().all()


def test_full_nan():
    aggregator = ConFIG()
    matrix = torch.full([10, 100], torch.nan)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_inf():
    aggregator = ConFIG()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.inf
    result = aggregator(matrix)
    assert result.eq(torch.full_like(result, torch.inf)).all()


def test_full_inf():
    aggregator = ConFIG()
    matrix = torch.full([10, 100], torch.inf)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_neg_inf():
    aggregator = ConFIG()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = -torch.inf
    result = aggregator(matrix)
    assert result.eq(torch.full_like(result, -torch.inf)).all()


def test_full_neg_inf():
    aggregator = ConFIG()
    matrix = torch.full([10, 100], -torch.inf)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_representations():
    A = ConFIG()
    assert repr(A) == "ConFIG(pref_vector=None)"
    assert str(A) == "ConFIG"

    A = ConFIG(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"))
    assert repr(A) == "ConFIG(pref_vector=tensor([1., 2., 3.]))"
    assert str(A) == "ConFIG([1., 2., 3.])"
