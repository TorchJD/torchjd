import torch
from pytest import mark
from torch import Tensor
from torch.testing import assert_close

from torchjd.aggregation import IMTLG

from ._asserts import (
    assert_expected_structure,
    assert_non_differentiable,
    assert_permutation_invariant,
)
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(IMTLG(), matrix) for matrix in scaled_matrices]
typical_pairs = [(IMTLG(), matrix) for matrix in typical_matrices]
requires_grad_pairs = [(IMTLG(), torch.ones(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: IMTLG, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: IMTLG, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: IMTLG, matrix: Tensor):
    assert_non_differentiable(aggregator, matrix)


def test_imtlg_zero():
    """
    Tests that IMTLG correctly returns the 0 vector in the special case where input matrix only
    consists of zeros.
    """

    A = IMTLG()
    J = torch.zeros(2, 3)
    assert_close(A(J), torch.zeros(3))


def test_representations():
    A = IMTLG()
    assert repr(A) == "IMTLG()"
    assert str(A) == "IMTLG"
