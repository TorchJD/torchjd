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


@mark.parametrize(
    ["value", "expectation"],
    [
        (0.0, does_not_raise()),
        (torch.nan, raises(ValueError)),
        (torch.inf, raises(ValueError)),
        (-torch.inf, raises(ValueError)),
    ],
)
def test_check_is_finite(value: float, expectation: ExceptionContext):
    matrix = torch.ones([5, 5])
    matrix[1, 2] = value
    with expectation:
        Aggregator._check_is_finite(matrix)
