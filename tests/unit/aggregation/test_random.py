import pytest

from torchjd.aggregation import Random

from ._property_testers import ExpectedStructureProperty


@pytest.mark.parametrize("aggregator", [Random()])
class TestRGW(ExpectedStructureProperty):
    pass


def test_representations():
    A = Random()
    assert repr(A) == "Random()"
    assert str(A) == "Random"
