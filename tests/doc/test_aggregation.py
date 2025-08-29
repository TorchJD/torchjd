"""This file contains the test corresponding to the usage example of Aggregator and Weighting."""

from torch.testing import assert_close


def test_aggregation_and_weighting():
    from torch import tensor

    from torchjd.aggregation import UPGrad, UPGradWeighting

    aggregator = UPGrad()
    jacobian = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    aggregation = aggregator(jacobian)

    assert_close(aggregation, tensor([0.2929, 1.9004, 1.9004]), rtol=0, atol=1e-4)

    weighting = UPGradWeighting()
    gramian = jacobian @ jacobian.T
    weights = weighting(gramian)

    assert_close(weights, tensor([1.1109, 0.7894]), rtol=0, atol=1e-4)
