import pytest
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import AlignedMTLWrapper, MeanWeighting, WeightedAggregator


@pytest.mark.parametrize(
    "aggregator",
    [WeightedAggregator(AlignedMTLWrapper(MeanWeighting()))],
)
class TestAlignedMTL(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    weighting = AlignedMTLWrapper(weighting=MeanWeighting())
    assert repr(weighting) == "AlignedMTLWrapper(weighting=MeanWeighting())"
    assert str(weighting) == "AlignedMTL MeanWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == (
        "WeightedAggregator(weighting=AlignedMTLWrapper(weighting=MeanWeighting" "()))"
    )
    assert str(aggregator) == "AlignedMTL Mean"
