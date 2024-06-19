import pytest
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import Mean


@pytest.mark.parametrize("aggregator", [Mean()])
class TestMean(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = Mean()
    assert repr(A) == "Mean()"
    assert str(A) == "Mean"
