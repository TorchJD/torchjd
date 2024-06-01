import pytest
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import DualProjWrapper, MeanWeighting, WeightedAggregator


@pytest.mark.parametrize(
    "aggregator",
    [WeightedAggregator(DualProjWrapper(MeanWeighting()))],
)
class TestDualProj(ExpectedShapeProperty, NonConflictingProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    weighting = DualProjWrapper(
        weighting=MeanWeighting(), norm_eps=0.0001, reg_eps=0.0001, solver="quadprog"
    )
    assert repr(weighting) == (
        "DualProjWrapper(weighting=MeanWeighting(), norm_eps=0.0001, "
        "reg_eps=0.0001, solver='quadprog')"
    )
    assert str(weighting) == "DualProj MeanWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == (
        "WeightedAggregator(weighting=DualProjWrapper(weighting="
        "MeanWeighting(), norm_eps=0.0001, reg_eps=0.0001, solver='quadprog'))"
    )
    assert str(aggregator) == "DualProj Mean"
