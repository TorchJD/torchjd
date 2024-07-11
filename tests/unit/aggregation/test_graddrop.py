import pytest
import torch

from torchjd.aggregation import GradDrop

from .utils import ExpectedShapeProperty


@pytest.mark.parametrize("aggregator", [GradDrop()])
class TestGradDrop(ExpectedShapeProperty):
    pass


def test_representations():
    A = GradDrop(leak=torch.tensor([0.0, 1.0]))
    assert repr(A) == "GradDrop(leak=tensor([0., 1.]))"
    assert str(A) == "GradDrop([0., 1.])"

    A = GradDrop()
    assert repr(A) == "GradDrop(leak=None)"
    assert str(A) == "GradDrop"
