import pytest

from torchjd.aggregation import AlignedMTL

from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@pytest.mark.parametrize("aggregator", [AlignedMTL()])
class TestAlignedMTL(ExpectedStructureProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = AlignedMTL(pref_vector=None)
    assert repr(A) == "AlignedMTL(pref_vector=None)"
    assert str(A) == "AlignedMTL"
