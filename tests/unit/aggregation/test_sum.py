from pytest import mark

from torchjd.aggregation import Sum

from ._property_testers import (
    ExpectedStructureProperty,
    LinearUnderScalingProperty,
    PermutationInvarianceProperty,
    StrongStationarityProperty,
)


@mark.parametrize("aggregator", [Sum()])
class TestSum(
    ExpectedStructureProperty,
    PermutationInvarianceProperty,
    LinearUnderScalingProperty,
    StrongStationarityProperty,
):
    pass


def test_representations():
    A = Sum()
    assert repr(A) == "Sum()"
    assert str(A) == "Sum"
