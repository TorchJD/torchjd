import pytest
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation import Random


@pytest.mark.parametrize("aggregator", [Random()])
class TestRGW(ExpectedShapeProperty):
    pass


def test_representations():
    A = Random()
    assert repr(A) == "Random()"
    assert str(A) == "Random"
