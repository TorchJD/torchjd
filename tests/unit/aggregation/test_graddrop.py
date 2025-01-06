import torch
from pytest import mark

from torchjd.aggregation import GradDrop

from ._property_testers import ExpectedStructureProperty


@mark.parametrize("aggregator", [GradDrop()])
class TestGradDrop(ExpectedStructureProperty):
    pass


def test_representations():
    A = GradDrop(leak=torch.tensor([0.0, 1.0], device="cpu"))
    assert repr(A) == "GradDrop(leak=tensor([0., 1.]))"
    assert str(A) == "GradDrop([0., 1.])"

    A = GradDrop()
    assert repr(A) == "GradDrop(leak=None)"
    assert str(A) == "GradDrop"
