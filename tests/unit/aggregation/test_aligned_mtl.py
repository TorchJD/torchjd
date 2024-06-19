import pytest
from unit.aggregation.utils.property_testers import (
    ExpectedShapeProperty,
    PermutationInvarianceProperty,
)

from torchjd.aggregation import AlignedMTL


@pytest.mark.parametrize("aggregator", [AlignedMTL()])
class TestAlignedMTL(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = AlignedMTL(pref_vector=None)
    assert repr(A) == "AlignedMTL(pref_vector=None)"
    assert str(A) == "AlignedMTL"
