import torch
from pytest import mark

from torchjd.aggregation import ConFIG

from ._property_testers import ExpectedStructureProperty


# For some reason, some permutation-invariance property tests fail with the pinv-based
# implementation.
@mark.parametrize("aggregator", [ConFIG()])
class TestConfig(ExpectedStructureProperty):
    pass


def test_representations():
    A = ConFIG()
    assert repr(A) == "ConFIG(pref_vector=None)"
    assert str(A) == "ConFIG"

    A = ConFIG(pref_vector=torch.tensor([1.0, 2.0, 3.0], device="cpu"))
    assert repr(A) == "ConFIG(pref_vector=tensor([1., 2., 3.]))"
    assert str(A) == "ConFIG([1., 2., 3.])"
