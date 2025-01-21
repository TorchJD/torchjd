import torch
from pytest import mark
from torch.testing import assert_close

from torchjd.aggregation._dual_cone_utils import _weights_of_projection_onto_dual_cone


@mark.parametrize("shape", [(5, 7), (9, 37), (2, 14), (32, 114), (50, 100)])
def test_lagrangian_satisfies_kkt_conditions(shape: tuple[int, int]):
    matrix = torch.randn(shape)
    weights = torch.rand(shape[0])

    gramian = matrix @ matrix.T

    projection_weights = _weights_of_projection_onto_dual_cone(gramian, weights, "quadprog")
    lagrange_multiplier = projection_weights - weights

    positive_lagrange_multiplier = lagrange_multiplier[lagrange_multiplier >= 0.0]
    assert_close(
        positive_lagrange_multiplier.norm(), lagrange_multiplier.norm(), atol=1e-05, rtol=0
    )

    constraint = gramian @ projection_weights

    positive_constraint = constraint[constraint >= 0]
    assert_close(positive_constraint.norm(), constraint.norm(), atol=1e-04, rtol=0)

    slackness = lagrange_multiplier @ constraint
    assert_close(slackness, torch.zeros_like(slackness), atol=3e-03, rtol=0)
