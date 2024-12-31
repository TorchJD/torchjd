from pytest import mark

from torchjd.aggregation import ConFIG

from ._property_testers import (
    ExpectedStructureProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
)


@mark.parametrize("aggregator", [ConFIG()])
class TestConFIG(ExpectedStructureProperty, NonConflictingProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = ConFIG()
    assert repr(A) == "ConFIG()"
    assert str(A) == "ConFIG"
