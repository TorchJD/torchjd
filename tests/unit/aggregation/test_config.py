from pytest import mark

from torchjd.aggregation import ConFIG

from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@mark.parametrize("aggregator", [ConFIG()])
class TestConFIG(ExpectedStructureProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = ConFIG(use_least_square=True)
    assert repr(A) == "ConFIG(use_least_square=True)"
    assert str(A) == "ConFIG"
