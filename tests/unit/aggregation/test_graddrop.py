import pytest
import torch
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation import GradDrop


@pytest.mark.parametrize("aggregator", [GradDrop()])
class TestGradDrop(ExpectedShapeProperty):
    pass


def test_representations():
    aggregator = GradDrop(leak=torch.tensor([0.0, 1.0]))
    assert repr(aggregator) == "GradDrop(leak=tensor([0., 1.]))"
    assert str(aggregator) == "GradDrop([0., 1.])"

    aggregator = GradDrop()
    assert repr(aggregator) == "GradDrop(leak=None)"
    assert str(aggregator) == "GradDrop"
