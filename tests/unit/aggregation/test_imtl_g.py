import pytest

from torchjd.aggregation import IMTLG

from ._property_testers import ExpectedShapeProperty, PermutationInvarianceProperty


@pytest.mark.parametrize("aggregator", [IMTLG()])
class TestIMTLG(ExpectedShapeProperty, PermutationInvarianceProperty):
    pass


def test_representations():
    A = IMTLG()
    assert repr(A) == "IMTLG()"
    assert str(A) == "IMTLG"
