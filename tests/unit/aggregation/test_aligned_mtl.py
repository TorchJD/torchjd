import torch
from pytest import mark

from torchjd.aggregation import AlignedMTL

from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@mark.parametrize("aggregator", [AlignedMTL()])
class TestAlignedMTL(ExpectedStructureProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = AlignedMTL(pref_vector=None)
    assert repr(A) == "AlignedMTL(pref_vector=None)"
    assert str(A) == "AlignedMTL"

    A = AlignedMTL(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"))
    assert repr(A) == "AlignedMTL(pref_vector=tensor([1., 2., 3.]))"
    assert str(A) == "AlignedMTL([1., 2., 3.])"
