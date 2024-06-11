from typing import Iterable

from torch import Tensor

from torchjd._transform._utils import _A
from torchjd._transform.base import Transform


class Identity(Transform[_A, _A]):
    def __init__(self, required_keys: Iterable[Tensor]):
        self._required_keys = set(required_keys)

    def _compute(self, tensor_dict: _A) -> _A:
        return tensor_dict

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._required_keys
