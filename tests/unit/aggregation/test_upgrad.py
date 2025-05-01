import torch
from pytest import mark

from torchjd.aggregation import UPGrad

from ._property_testers import (
    ExpectedStructureProperty,
    LinearUnderScalingProperty,
    NonConflictingProperty,
    NonDifferentiableProperty,
    PermutationInvarianceProperty,
    StrongStationarityProperty,
)


@mark.parametrize("aggregator", [UPGrad()])
class TestUPGrad(
    ExpectedStructureProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
    LinearUnderScalingProperty,
    StrongStationarityProperty,
    NonDifferentiableProperty,
):
    pass


def test_representations():
    A = UPGrad(pref_vector=None, max_iter=100, eps=0.0001)
    assert repr(A) == "UPGrad(pref_vector=None, max_iter=100, eps=0.0001)"
    assert str(A) == "UPGrad"

    A = UPGrad(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"), max_iter=100, eps=0.0001)
    assert repr(A) == "UPGrad(pref_vector=tensor([1., 2., 3.]), max_iter=100, eps=0.0001)"
    assert str(A) == "UPGrad([1., 2., 3.])"
