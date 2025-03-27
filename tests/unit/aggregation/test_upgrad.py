import torch
from pytest import mark

from torchjd.aggregation import UPGrad

from ._property_testers import (
    ExpectedStructureProperty,
    LinearUnderScalingProperty,
    NonConflictingProperty,
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
):
    pass


def test_representations():
    A = UPGrad(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver="quadprog")
    assert repr(A) == "UPGrad(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver='quadprog')"
    assert str(A) == "UPGrad"

    A = UPGrad(
        pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"),
        norm_eps=0.0001,
        reg_eps=0.0001,
        solver="quadprog",
    )
    assert (
        repr(A) == "UPGrad(pref_vector=tensor([1., 2., 3.]), norm_eps=0.0001, reg_eps=0.0001, "
        "solver='quadprog')"
    )
    assert str(A) == "UPGrad([1., 2., 3.])"
