import pytest

from torchjd.aggregation import Random

from ._property_testers import ExpectedShapeProperty


@pytest.mark.parametrize("aggregator", [Random()])
class TestRGW(ExpectedShapeProperty):
    pass


def test_representations():
    A = Random()
    assert repr(A) == "Random()"
    assert str(A) == "Random"
