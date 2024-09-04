from contextlib import nullcontext as does_not_raise
from typing import Sequence

import pytest
import torch
from unit._utils import ExceptionContext
from unit.conftest import DEVICE

from torchjd.aggregation import Aggregator


@pytest.mark.parametrize(
    ["shape", "expectation"],
    [
        ([], pytest.raises(ValueError)),
        ([1], pytest.raises(ValueError)),
        ([1, 2], does_not_raise()),
        ([1, 2, 3], pytest.raises(ValueError)),
        ([1, 2, 3, 4], pytest.raises(ValueError)),
    ],
)
def test_check_is_matrix(shape: Sequence[int], expectation: ExceptionContext):
    with expectation:
        Aggregator._check_is_matrix(torch.randn(shape, device=DEVICE))
