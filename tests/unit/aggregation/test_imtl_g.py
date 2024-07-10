import pytest

from torchjd.aggregation import IMTLG

from .utils import ExpectedShapeProperty, PermutationInvarianceProperty


@pytest.mark.parametrize("aggregator", [IMTLG()])
class TestIMTLG(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = IMTLG()
    assert repr(A) == "IMTLG()"
    assert str(A) == "IMTLG"
