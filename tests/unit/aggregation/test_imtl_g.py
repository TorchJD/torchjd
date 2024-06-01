import pytest
from unit.aggregation.utils import ExpectedShapeProperty, PermutationInvarianceProperty

from torchjd.aggregation import IMTLGWeighting, WeightedAggregator


@pytest.mark.parametrize("aggregator", [WeightedAggregator(IMTLGWeighting())])
class TestIMTLG(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    weighting = IMTLGWeighting()
    assert repr(weighting) == "IMTLGWeighting()"
    assert str(weighting) == "IMTLGWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == "WeightedAggregator(weighting=IMTLGWeighting())"
    assert str(aggregator) == "IMTLG"
