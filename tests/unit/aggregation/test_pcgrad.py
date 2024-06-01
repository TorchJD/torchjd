import pytest
import torch
from torch.testing import assert_close
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation import PCGradWeighting, SumWeighting, UPGradWrapper, WeightedAggregator


@pytest.mark.parametrize("aggregator", [WeightedAggregator(PCGradWeighting())])
class TestPCGrad(ExpectedShapeProperty):
    pass


@pytest.mark.parametrize(
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
    Tests that UPGradWrapper of a SumWeighting is equivalent to PCGradWeighting for matrices of 2
    rows.
    """

    matrix = torch.randn(shape)

    pc_grad_weighting = PCGradWeighting()
    upgrad_sum_weighting = UPGradWrapper(SumWeighting())

    result = pc_grad_weighting(matrix)
    expected = upgrad_sum_weighting(matrix)

    assert_close(result, expected, atol=4e-04, rtol=0.0)


def test_representations():
    weighting = PCGradWeighting()
    assert repr(weighting) == "PCGradWeighting()"
    assert str(weighting) == "PCGradWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == "WeightedAggregator(weighting=PCGradWeighting())"
    assert str(aggregator) == "PCGrad"
