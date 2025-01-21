import torch
from pytest import mark, raises
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


@mark.parametrize("shape", [(5, 7), (9, 37), (2, 14), (32, 114), (50, 100)])
def test_lagrangian_satisfies_kkt_conditions_matrix_weights(shape: tuple[int, int]):
    matrix = torch.randn(shape)
    weights_matrix = torch.diag(torch.rand(shape[0]))

    gramian = matrix @ matrix.T

    projection_weights = _weights_of_projection_onto_dual_cone(gramian, weights_matrix, "quadprog")
    lagrange_multiplier = projection_weights - weights_matrix

    positive_lagrange_multiplier = lagrange_multiplier[lagrange_multiplier >= 0.0]
    assert_close(
        positive_lagrange_multiplier.norm(), lagrange_multiplier.norm(), atol=1e-05, rtol=0
    )

    constraint = gramian @ projection_weights

    positive_constraint = constraint[constraint >= 0]
    assert_close(positive_constraint.norm(), constraint.norm(), atol=1e-04, rtol=0)

    slackness = torch.trace(constraint @ lagrange_multiplier.T)
    assert_close(slackness, torch.zeros_like(slackness), atol=3e-03, rtol=0)


def test_weights_of_projection_onto_dual_cone_invalid_shape():
    with raises(ValueError):
        _weights_of_projection_onto_dual_cone(
            torch.zeros([5, 5]), torch.zeros([5, 2, 3]), "quadprog"
        )
