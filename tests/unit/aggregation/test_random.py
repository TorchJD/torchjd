from pytest import mark

from torchjd.aggregation import Random

from ._property_testers import ExpectedStructureProperty, LinearUnderScalingProperty


@mark.parametrize("aggregator", [Random()])
class TestRGW(ExpectedStructureProperty, LinearUnderScalingProperty):
    pass


def test_representations():
    A = Random()
    assert repr(A) == "Random()"
    assert str(A) == "Random"
