import torch
from pytest import mark

from torchjd.aggregation import DualProj

from ._property_testers import (
    ExpectedStructureProperty,
    NonConflictingProperty,
    NonDifferentiableProperty,
    PermutationInvarianceProperty,
    StrongStationarityProperty,
)


@mark.parametrize("aggregator", [DualProj()])
class TestDualProj(
    ExpectedStructureProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
    StrongStationarityProperty,
    NonDifferentiableProperty,
):
    pass


def test_representations():
    A = DualProj(pref_vector=None, max_iter=100, eps=0.0001)
    assert repr(A) == "DualProj(pref_vector=None, max_iter=100, eps=0.0001)"
    assert str(A) == "DualProj"

    A = DualProj(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"), max_iter=100, eps=0.0001)
    assert repr(A) == "DualProj(pref_vector=tensor([1., 2., 3.]), max_iter=100, eps=0.0001)"
    assert str(A) == "DualProj([1., 2., 3.])"
