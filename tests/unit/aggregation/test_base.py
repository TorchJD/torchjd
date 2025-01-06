from contextlib import nullcontext as does_not_raise
from typing import Sequence

import torch
from pytest import mark, raises
from unit._utils import ExceptionContext

from torchjd.aggregation import Aggregator


@mark.parametrize(
    ["shape", "expectation"],
    [
        ([], raises(ValueError)),
        ([1], raises(ValueError)),
        ([1, 2], does_not_raise()),
        ([1, 2, 3], raises(ValueError)),
        ([1, 2, 3, 4], raises(ValueError)),
    ],
)
def test_check_is_matrix(shape: Sequence[int], expectation: ExceptionContext):
    with expectation:
        Aggregator._check_is_matrix(torch.randn(shape))
