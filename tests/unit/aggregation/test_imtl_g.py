import pytest
from unit.aggregation.utils import ExpectedShapeProperty, PermutationInvarianceProperty

from torchjd.aggregation import IMTLG


@pytest.mark.parametrize("aggregator", [IMTLG()])
class TestIMTLG(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = IMTLG()
    assert repr(A) == "IMTLG()"
    assert str(A) == "IMTLG"
