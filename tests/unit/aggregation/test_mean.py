from pytest import mark

from torchjd.aggregation import Mean

from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@mark.parametrize("aggregator", [Mean()])
class TestMean(ExpectedStructureProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = Mean()
    assert repr(A) == "Mean()"
    assert str(A) == "Mean"
