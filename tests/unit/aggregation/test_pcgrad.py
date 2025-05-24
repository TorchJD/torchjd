import torch
from pytest import mark
from torch import Tensor
from torch.testing import assert_close

from torchjd.aggregation import PCGrad
from torchjd.aggregation._pcgrad import _PCGradWeighting
from torchjd.aggregation._sum import _SumWeighting
from torchjd.aggregation._upgrad import _UPGradWrapper
from torchjd.aggregation._utils.gramian import compute_gramian

from ._asserts import assert_expected_structure, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(PCGrad(), matrix) for matrix in scaled_matrices]
typical_pairs = [(PCGrad(), matrix) for matrix in typical_matrices]
requires_grad_pairs = [(PCGrad(), torch.ones(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: PCGrad, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: PCGrad, matrix: Tensor):
    assert_non_differentiable(aggregator, matrix)


@mark.parametrize(
    "shape",
    [
        (2, 5),
        (2, 7),
        (2, 9),
        (2, 15),
        (2, 27),
        (2, 68),
        (2, 102),
        (2, 57),
        (2, 1200),
        (2, 11100),
    ],
)
def test_equivalence_upgrad_sum_two_rows(shape: tuple[int, int]):
    """
    Tests that _UPGradWrapper of a _SumWeighting is equivalent to _PCGradWeighting for matrices of 2
    rows.
    """

    matrix = torch.randn(shape)
    gramian = compute_gramian(matrix)

    pc_grad_weighting = _PCGradWeighting()
    upgrad_sum_weighting = _UPGradWrapper(
        _SumWeighting(), norm_eps=0.0, reg_eps=0.0, solver="quadprog"
    )

    result = pc_grad_weighting(gramian)
    expected = upgrad_sum_weighting(gramian)

    assert_close(result, expected, atol=4e-04, rtol=0.0)


def test_representations():
    A = PCGrad()
    assert repr(A) == "PCGrad()"
    assert str(A) == "PCGrad"
