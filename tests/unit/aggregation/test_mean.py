import pytest

from torchjd.aggregation import Mean

from ._property_testers import ExpectedShapeProperty, PermutationInvarianceProperty


@pytest.mark.parametrize("aggregator", [Mean()])
class TestMean(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = Mean()
    assert repr(A) == "Mean()"
    assert str(A) == "Mean"
