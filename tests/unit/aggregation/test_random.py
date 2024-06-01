import pytest
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation import RandomWeighting, WeightedAggregator


@pytest.mark.parametrize("aggregator", [WeightedAggregator(RandomWeighting())])
class TestRGW(ExpectedShapeProperty):
    pass


def test_representations():
    weighting = RandomWeighting()
    assert repr(weighting) == "RandomWeighting()"
    assert str(weighting) == "RandomWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == "WeightedAggregator(weighting=RandomWeighting())"
    assert str(aggregator) == "Random"
