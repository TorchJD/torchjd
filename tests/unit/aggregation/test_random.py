from pytest import mark

from torchjd.aggregation import Random

from ._property_testers import ExpectedStructureProperty, StrongStationarityProperty


@mark.parametrize("aggregator", [Random()])
class TestRandom(ExpectedStructureProperty, StrongStationarityProperty):
    pass


def test_representations():
    A = Random()
    assert repr(A) == "Random()"
    assert str(A) == "Random"
