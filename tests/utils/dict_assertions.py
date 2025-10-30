from collections.abc import Hashable
from typing import TypeVar

from torch import Tensor
from torch.testing import assert_close

_KeyType = TypeVar("_KeyType", bound=Hashable)


def assert_tensor_dicts_are_close(d1: dict[_KeyType, Tensor], d2: dict[_KeyType, Tensor]) -> None:
    """
    Check that two dictionaries of tensors are close enough. Note that this does not require the
    keys to have the same ordering.
    """

    assert d1.keys() == d2.keys()

    for key in d1:
        assert_close(d1[key], d2[key])
