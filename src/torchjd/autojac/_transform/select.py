from typing import Iterable

from torch import Tensor

from ._utils import _A
from .base import Transform


class Select(Transform[_A, _A]):
    def __init__(self, keys: Iterable[Tensor], required_keys: Iterable[Tensor]):
        self.keys = set(keys)
        self._required_keys = set(required_keys)

    def __call__(self, tensor_dict: _A) -> _A:
        output = {key: tensor_dict[key] for key in self.keys}
        return type(tensor_dict)(output)

    def check_and_get_keys(self) -> tuple[set[Tensor], set[Tensor]]:
        required_keys = set(self._required_keys)
        if not self.keys.issubset(required_keys):
            raise ValueError("Parameter `keys` should be a subset of parameter `required_keys`")

        return required_keys, self.keys
