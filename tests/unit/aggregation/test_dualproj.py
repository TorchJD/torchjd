import torch
from pytest import mark

from torchjd.aggregation import DualProj

from ._property_testers import (
    ExpectedStructureProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
)


@mark.parametrize("aggregator", [DualProj()])
class TestDualProj(
    ExpectedStructureProperty, NonConflictingProperty, PermutationInvarianceProperty
):
    pass


def test_representations():
    A = DualProj(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver="quadprog")
    assert (
        repr(A) == "DualProj(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver='quadprog')"
    )
    assert str(A) == "DualProj"

    A = DualProj(
        pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"),
        norm_eps=0.0001,
        reg_eps=0.0001,
        solver="quadprog",
    )
    assert (
        repr(A) == "DualProj(pref_vector=tensor([1., 2., 3.]), norm_eps=0.0001, reg_eps=0.0001, "
        "solver='quadprog')"
    )
    assert str(A) == "DualProj([1., 2., 3.])"
