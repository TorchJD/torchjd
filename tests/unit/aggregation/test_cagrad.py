from contextlib import nullcontext as does_not_raise

from pytest import mark, raises
from unit._utils import ExceptionContext

from torchjd.aggregation import CAGrad

from ._property_testers import ExpectedStructureProperty, NonConflictingProperty


@mark.parametrize("aggregator", [CAGrad(c=0.5)])
class TestCAGrad(ExpectedStructureProperty):
    pass


@mark.parametrize("aggregator", [CAGrad(c=1.0), CAGrad(c=2.0)])
class TestCAGradNonConflicting(NonConflictingProperty):
    """Tests that CAGrad is non-conflicting when c >= 1 (it should not hold when c < 1)"""

    pass


@mark.parametrize(
    ["c", "expectation"],
    [
        (-5.0, raises(ValueError)),
        (-1.0, raises(ValueError)),
        (0.0, does_not_raise()),
        (1.0, does_not_raise()),
        (50.0, does_not_raise()),
    ],
)
def test_c_check(c: float, expectation: ExceptionContext):
    with expectation:
        _ = CAGrad(c=c)


def test_representations():
    A = CAGrad(c=0.5, norm_eps=0.0001)
    assert repr(A) == "CAGrad(c=0.5, norm_eps=0.0001)"
    assert str(A) == "CAGrad0.5"
