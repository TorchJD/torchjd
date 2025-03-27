import torch
from pytest import mark
from torch.testing import assert_close

from torchjd.aggregation import MGDA
from torchjd.aggregation.mgda import _MGDAWeighting

from ._property_testers import (
    ExpectedStructureProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
)


@mark.parametrize("aggregator", [MGDA()])
class TestMGDA(ExpectedStructureProperty, NonConflictingProperty, PermutationInvarianceProperty):
    pass


@mark.parametrize(
    "shape",
    [
        (5, 7),
        (9, 37),
        (2, 14),
        (32, 114),
        (50, 100),
    ],
)
def test_mgda_satisfies_kkt_conditions(shape: tuple[int, int]):
    matrix = torch.randn(shape)
    weighting = _MGDAWeighting(epsilon=1e-05, max_iters=1000)

    gramian = matrix @ matrix.T

    weights = weighting(matrix)

    output_direction = gramian @ weights  # Stationarity
    lamb = -weights @ output_direction  # Complementary slackness
    mu = output_direction + lamb

    # Primal feasibility
    positive_weights = weights[weights >= 0]
    assert_close(positive_weights.norm(), weights.norm())

    weights_sum = weights.sum()
    assert_close(weights_sum, torch.ones([]))

    # Dual feasibility
    positive_mu = mu[mu >= 0]
    assert_close(positive_mu.norm(), mu.norm(), atol=1e-02, rtol=0.0)


def test_representations():
    A = MGDA(epsilon=0.001, max_iters=100)
    assert repr(A) == "MGDA(epsilon=0.001, max_iters=100)"
    assert str(A) == "MGDA"
