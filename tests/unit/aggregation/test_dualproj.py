import pytest

from torchjd.aggregation import DualProj

from .utils import ExpectedShapeProperty, NonConflictingProperty, PermutationInvarianceProperty


@pytest.mark.parametrize("aggregator", [DualProj()])
class TestDualProj(ExpectedShapeProperty, NonConflictingProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = DualProj(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver="quadprog")
    assert (
        repr(A) == "DualProj(pref_vector=None, norm_eps=0.0001, reg_eps=0.0001, solver='quadprog')"
    )
    assert str(A) == "DualProj"
