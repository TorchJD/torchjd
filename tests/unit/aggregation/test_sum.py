import pytest
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import SumWeighting, WeightedAggregator


@pytest.mark.parametrize("aggregator", [WeightedAggregator(SumWeighting())])
class TestSum(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    weighting = SumWeighting()
    assert repr(weighting) == "SumWeighting()"
    assert str(weighting) == "SumWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == "WeightedAggregator(weighting=SumWeighting())"
    assert str(aggregator) == "Sum"
