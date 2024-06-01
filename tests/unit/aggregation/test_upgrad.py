import pytest
import torch
from torch.testing import assert_close
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import MeanWeighting, UPGradWrapper, WeightedAggregator


@pytest.mark.parametrize(
    "aggregator",
    [WeightedAggregator(UPGradWrapper(MeanWeighting()))],
)
class TestUPGrad(ExpectedShapeProperty, NonConflictingProperty, PermutationInvarianceProperty):
    pass


@pytest.mark.parametrize(
    "shape",
    [
        (5, 7),
        (9, 37),
        (2, 14),
        (32, 114),
        (50, 100),
    ],
)
def test_upgrad_lagrangian_satisfies_kkt_conditions(shape: tuple[int, int]):
    matrix = torch.randn(shape)
    weights = torch.rand(shape[0])

    gramian = matrix @ matrix.T

    upgrad = UPGradWrapper(MeanWeighting(), norm_eps=0.0001, reg_eps=0.0, solver="quadprog")

    lagrange_multiplier = upgrad._compute_lagrangian(matrix, weights)

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
    weighting = UPGradWrapper(
        weighting=MeanWeighting(), norm_eps=0.0001, reg_eps=0.0001, solver="quadprog"
    )
    assert repr(weighting) == (
        "UPGradWrapper(weighting=MeanWeighting(), norm_eps=0.0001, "
        "reg_eps=0.0001, solver='quadprog')"
    )
    assert str(weighting) == "UPGrad MeanWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == (
        "WeightedAggregator(weighting=UPGradWrapper(weighting=MeanWeighting(), "
        "norm_eps=0.0001, reg_eps=0.0001, solver='quadprog'))"
    )
    assert str(aggregator) == "UPGrad Mean"
