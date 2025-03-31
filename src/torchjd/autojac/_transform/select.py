from typing import Iterable

from torch import Tensor

from ._utils import _A
from .base import RequirementError, Transform


class Select(Transform[_A, _A]):
    def __init__(self, keys: Iterable[Tensor]):
        self.keys = set(keys)

    def __call__(self, tensor_dict: _A) -> _A:
        output = {key: tensor_dict[key] for key in self.keys}
        return type(tensor_dict)(output)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        if not self.keys.issubset(input_keys):
            raise RequirementError(
                f"The input_keys needs to be a super set of the keys to select. Found {input_keys} "
                f"and {self.keys}"
            )
        return self.keys
