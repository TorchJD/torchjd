import torch
from pytest import mark
from torch import Tensor, tensor
from torch.testing import assert_close
from utils.tensors import ones_

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
requires_grad_pairs = [(ConFIG(), ones_(3, 5, requires_grad=True))]


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


def test_value():
    """Test that the output values are fixed (on cpu)."""

    A = ConFIG()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    assert_close(A(J), tensor([0.1588, 2.0706, 2.0706]), rtol=0, atol=1e-4)


def test_representations():
    A = ConFIG()
    assert repr(A) == "ConFIG(pref_vector=None)"
    assert str(A) == "ConFIG"

    A = ConFIG(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"))
    assert repr(A) == "ConFIG(pref_vector=tensor([1., 2., 3.]))"
    assert str(A) == "ConFIG([1., 2., 3.])"
