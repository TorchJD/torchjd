import torch
from pytest import mark
from torch.testing import assert_close
from unit.conftest import DEVICE

from torchjd.aggregation import IMTLG

from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@mark.parametrize("aggregator", [IMTLG()])
class TestIMTLG(ExpectedStructureProperty, PermutationInvarianceProperty):
    pass


def test_imtlg_zero():
    """
    Tests that IMTLG correctly returns the 0 vector in the special case where input matrix only
    consists of zeros.
    """

    A = IMTLG()
    J = torch.zeros(2, 3, device=DEVICE)

    aggregation = A(J)
    expected = torch.zeros(3, device=DEVICE)

    assert_close(aggregation, expected)


def test_representations():
    A = IMTLG()
    assert repr(A) == "IMTLG()"
    assert str(A) == "IMTLG"
