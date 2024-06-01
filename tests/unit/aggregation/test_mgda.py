import pytest
import torch
from torch.testing import assert_close
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    NonConflictingProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import MGDAWeighting, WeightedAggregator


@pytest.mark.parametrize("aggregator", [WeightedAggregator(MGDAWeighting())])
class TestMGDA(ExpectedShapeProperty, NonConflictingProperty, PermutationInvarianceProperty):
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
def test_mgda_satisfies_kkt_conditions(shape: tuple[int, int]):
    matrix = torch.randn(shape)
    weighting = MGDAWeighting(epsilon=1e-05, max_iters=10000)

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
    assert_close(positive_mu.norm(), mu.norm(), atol=3e-04, rtol=0.0)


def test_representations():
    weighting = MGDAWeighting(epsilon=0.001, max_iters=100)
    assert repr(weighting) == "MGDAWeighting(epsilon=0.001, max_iters=100)"
    assert str(weighting) == "MGDAWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == (
        "WeightedAggregator(weighting=MGDAWeighting(epsilon=0.001, max_iters=100))"
    )
    assert str(aggregator) == "MGDA"
