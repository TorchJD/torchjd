import pytest
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation import (
    MeanWeighting,
    MGDAWeighting,
    NormalizingWrapper,
    SumWeighting,
    WeightedAggregator,
)


@pytest.mark.parametrize(
    "aggregator",
    [
        WeightedAggregator(NormalizingWrapper(SumWeighting(), norm_p=0.5, norm_value=1.0)),
        WeightedAggregator(NormalizingWrapper(SumWeighting(), norm_p=1.0, norm_value=1.0)),
        WeightedAggregator(NormalizingWrapper(MeanWeighting(), norm_p=2.0, norm_value=1.0)),
        WeightedAggregator(NormalizingWrapper(MGDAWeighting(), norm_p=10.0, norm_value=1.0)),
    ],
)
class TestNormalizing(ExpectedShapeProperty):
    pass


def test_representations():
    weighting = NormalizingWrapper(MeanWeighting(), norm_p=2.0, norm_value=1.0, norm_eps=0.0)
    assert repr(weighting) == (
        "NormalizingWrapper(weighting=MeanWeighting(), norm_p=2.0, norm_value=1.0, " "norm_eps=0.0)"
    )
    assert str(weighting) == "Norm2.0-1.0 MeanWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == (
        "WeightedAggregator(weighting=NormalizingWrapper(weighting="
        "MeanWeighting(), norm_p=2.0, norm_value=1.0, norm_eps=0.0))"
    )
    assert str(aggregator) == "Norm2.0-1.0 Mean"
