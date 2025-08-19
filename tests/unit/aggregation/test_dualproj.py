import torch
from pytest import mark
from torch import Tensor, tensor
from torch.testing import assert_close
from utils.tensors import ones_

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
requires_grad_pairs = [(DualProj(), ones_(3, 5, requires_grad=True))]


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


def test_value():
    """Test that the output values are fixed (on cpu)."""

    A = DualProj()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    assert_close(A(J), tensor([0.5563, 1.1109, 1.1109]), rtol=0, atol=1e-4)


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
