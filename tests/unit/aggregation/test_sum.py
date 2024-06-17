import pytest
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import Sum


@pytest.mark.parametrize("aggregator", [Sum()])
class TestSum(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = Sum()
    assert repr(A) == "Sum()"
    assert str(A) == "Sum"
