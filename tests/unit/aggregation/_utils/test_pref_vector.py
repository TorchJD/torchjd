from contextlib import nullcontext as does_not_raise

from pytest import mark, raises
from torch import Tensor
from unit._utils import ExceptionContext, ones_

from torchjd.aggregation._mean import _MeanWeighting
from torchjd.aggregation._utils.pref_vector import pref_vector_to_weighting


@mark.parametrize(
    ["pref_vector", "expectation"],
    [
        (None, does_not_raise()),
        (ones_([]), raises(ValueError)),
        (ones_([0]), does_not_raise()),
        (ones_([1]), does_not_raise()),
        (ones_([5]), does_not_raise()),
        (ones_([1, 1]), raises(ValueError)),
        (ones_([1, 1, 1]), raises(ValueError)),
    ],
)
def test_pref_vector_to_weighting_check(pref_vector: Tensor | None, expectation: ExceptionContext):
    with expectation:
        _ = pref_vector_to_weighting(pref_vector, default=_MeanWeighting())
