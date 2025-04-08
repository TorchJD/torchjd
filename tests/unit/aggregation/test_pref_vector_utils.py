from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch import Tensor
from unit._utils import ExceptionContext

from torchjd.aggregation._pref_vector_utils import pref_vector_to_weighting
from torchjd.aggregation.mean import _MeanWeighting


@mark.parametrize(
    ["pref_vector", "expectation"],
    [
        (None, does_not_raise()),
        (torch.ones([]), raises(ValueError)),
        (torch.ones([0]), does_not_raise()),
        (torch.ones([1]), does_not_raise()),
        (torch.ones([5]), does_not_raise()),
        (torch.ones([1, 1]), raises(ValueError)),
        (torch.ones([1, 1, 1]), raises(ValueError)),
    ],
)
def test_pref_vector_to_weighting_check(pref_vector: Tensor | None, expectation: ExceptionContext):
    with expectation:
        _ = pref_vector_to_weighting(pref_vector, default=_MeanWeighting())
