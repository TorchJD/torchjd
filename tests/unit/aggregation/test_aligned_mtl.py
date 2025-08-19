import torch
from pytest import mark
from torch import Tensor, tensor
from torch.testing import assert_close

from torchjd.aggregation import AlignedMTL

from ._asserts import assert_expected_structure, assert_permutation_invariant
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(AlignedMTL(), matrix) for matrix in scaled_matrices]
typical_pairs = [(AlignedMTL(), matrix) for matrix in typical_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: AlignedMTL, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: AlignedMTL, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


def test_value():
    """Test that the output values are fixed (on cpu)."""

    A = AlignedMTL()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    assert_close(A(J), tensor([0.2133, 0.9673, 0.9673]), rtol=0, atol=1e-4)


def test_representations():
    A = AlignedMTL(pref_vector=None)
    assert repr(A) == "AlignedMTL(pref_vector=None)"
    assert str(A) == "AlignedMTL"

    A = AlignedMTL(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"))
    assert repr(A) == "AlignedMTL(pref_vector=tensor([1., 2., 3.]))"
    assert str(A) == "AlignedMTL([1., 2., 3.])"
