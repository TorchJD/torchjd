import torch
from pytest import mark
from torch.testing import assert_close

from torchjd.aggregation import UPGrad
from torchjd.aggregation.mean import _MeanWeighting
from torchjd.aggregation.upgrad import _UPGradWrapper

from ._property_testers import (
    ExpectedStructureProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
)


@mark.parametrize("aggregator", [UPGrad()])
class TestUPGrad(ExpectedStructureProperty, NonConflictingProperty, PermutationInvarianceProperty):
    pass


@mark.parametrize("shape", [(5, 7), (9, 37), (2, 14), (32, 114), (50, 100)])
def test_upgrad_lagrangian_satisfies_kkt_conditions(shape: tuple[int, int]):
    matrix = torch.randn(shape)
    weights = torch.rand(shape[0])

    gramian = matrix @ matrix.T

    W = _UPGradWrapper(_MeanWeighting(), norm_eps=0.0001, reg_eps=0.0, solver="quadprog")

    lagrange_multiplier = W._compute_lagrangian(matrix, weights)

    positive_lagrange_multiplier = lagrange_multiplier[lagrange_multiplier >= 0]
    assert_close(
        positive_lagrange_multiplier.norm(), lagrange_multiplier.norm(), atol=1e-05, rtol=0
    )

    constraint = gramian @ (torch.diag(weights) + lagrange_multiplier.T)

    positive_constraint = constraint[constraint >= 0]
    assert_close(positive_constraint.norm(), constraint.norm(), atol=1e-04, rtol=0)

    slackness = torch.trace(lagrange_multiplier @ constraint)
    assert_close(slackness, torch.zeros_like(slackness), atol=3e-03, rtol=0)


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
