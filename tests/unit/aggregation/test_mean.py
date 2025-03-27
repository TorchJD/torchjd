from pytest import mark

from torchjd.aggregation import Mean

from ._property_testers import (
    ExpectedStructureProperty,
    LinearUnderScalingProperty,
    PermutationInvarianceProperty,
    StrongStationarityProperty,
)


@mark.parametrize("aggregator", [Mean()])
class TestMean(
    ExpectedStructureProperty,
    PermutationInvarianceProperty,
    LinearUnderScalingProperty,
    StrongStationarityProperty,
):
    pass


def test_representations():
    A = Mean()
    assert repr(A) == "Mean()"
    assert str(A) == "Mean"
