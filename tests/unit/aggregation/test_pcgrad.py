import torch
from pytest import mark
from torch.testing import assert_close

from torchjd.aggregation import PCGrad
from torchjd.aggregation.pcgrad import _PCGradWeighting
from torchjd.aggregation.sum import _SumWeighting
from torchjd.aggregation.upgrad import _UPGradWrapper

from ._property_testers import ExpectedStructureProperty


@mark.parametrize("aggregator", [PCGrad()])
class TestPCGrad(ExpectedStructureProperty):
    pass


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

    pc_grad_weighting = _PCGradWeighting()
    upgrad_sum_weighting = _UPGradWrapper(
        _SumWeighting(), norm_eps=0.0, reg_eps=0.0, solver="quadprog"
    )

    result = pc_grad_weighting(matrix)
    expected = upgrad_sum_weighting(matrix)

    assert_close(result, expected, atol=4e-04, rtol=0.0)


def test_representations():
    A = PCGrad()
    assert repr(A) == "PCGrad()"
    assert str(A) == "PCGrad"
