import torch
from pytest import mark
from torch import Tensor

from torchjd.aggregation import DualProj

from ._asserts import (
    assert_expected_structure,
    assert_non_conflicting,
    assert_non_differentiable,
    assert_permutation_invariant,
    assert_strongly_stationary,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(DualProj(), matrix) for matrix in scaled_matrices]
typical_pairs = [(DualProj(), matrix) for matrix in typical_matrices]
non_strong_pairs = [(DualProj(), matrix) for matrix in non_strong_matrices]
requires_grad_pairs = [(DualProj(), torch.ones(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: DualProj, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_non_conflicting(aggregator: DualProj, matrix: Tensor):
    assert_non_conflicting(aggregator, matrix, atol=5e-05, rtol=5e-05)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: DualProj, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix, n_runs=5, atol=2e-07, rtol=2e-07)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: DualProj, matrix: Tensor):
    assert_strongly_stationary(aggregator, matrix, threshold=3e-03)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: DualProj, matrix: Tensor):
    assert_non_differentiable(aggregator, matrix)


def test_one_nan():
    aggregator = DualProj()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.nan
    result = aggregator(matrix)
    assert result.isnan().all()


def test_full_nan():
    aggregator = DualProj()
    matrix = torch.full([10, 100], torch.nan)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_inf():
    aggregator = DualProj()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.inf
    result = aggregator(matrix)
    assert result.isnan().all()


def test_full_inf():
    aggregator = DualProj()
    matrix = torch.full([10, 100], torch.inf)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_neg_inf():
    aggregator = DualProj()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = -torch.inf
    result = aggregator(matrix)
    assert result.isnan().all()


def test_full_neg_inf():
    aggregator = DualProj()
    matrix = torch.full([10, 100], -torch.inf)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_representations():
    A = DualProj(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver="quadprog")
    assert (
        repr(A) == "DualProj(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver='quadprog')"
    )
    assert str(A) == "DualProj"

    A = DualProj(
        pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"),
        norm_eps=0.0001,
        reg_eps=0.0001,
        solver="quadprog",
    )
    assert (
        repr(A) == "DualProj(pref_vector=tensor([1., 2., 3.]), norm_eps=0.0001, reg_eps=0.0001, "
        "solver='quadprog')"
    )
    assert str(A) == "DualProj([1., 2., 3.])"
