import pytest
import torch
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation import GradDropAggregator


@pytest.mark.parametrize("aggregator", [GradDropAggregator()])
class TestGradDrop(ExpectedShapeProperty):
    pass


def test_representations():
    aggregator = GradDropAggregator(leak=torch.tensor([0.0, 1.0]))
    assert repr(aggregator) == "GradDropAggregator(leak=tensor([0., 1.]))"
    assert str(aggregator) == "GradDrop([0., 1.])"

    aggregator = GradDropAggregator()
    assert repr(aggregator) == "GradDropAggregator(leak=None)"
    assert str(aggregator) == "GradDrop"
