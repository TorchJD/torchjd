import pytest

from torchjd.aggregation import Sum

from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@pytest.mark.parametrize("aggregator", [Sum()])
class TestSum(ExpectedStructureProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = Sum()
    assert repr(A) == "Sum()"
    assert str(A) == "Sum"
