import pytest
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import MeanWeighting, WeightedAggregator


@pytest.mark.parametrize("aggregator", [WeightedAggregator(MeanWeighting())])
class TestMean(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    weighting = MeanWeighting()
    assert repr(weighting) == "MeanWeighting()"
    assert str(weighting) == "MeanWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == "WeightedAggregator(weighting=MeanWeighting())"
    assert str(aggregator) == "Mean"
