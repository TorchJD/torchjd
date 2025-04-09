from collections.abc import Set

from torch import Tensor

from .base import RequirementError, Transform
from .tensor_dict import _A


class Select(Transform[_A, _A]):
    def __init__(self, keys: Set[Tensor]):
        self.keys = keys

    def __call__(self, tensor_dict: _A) -> _A:
        output = {key: tensor_dict[key] for key in self.keys}
        return type(tensor_dict)(output)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        keys = set(self.keys)
        if not keys.issubset(input_keys):
            raise RequirementError(
                f"The input_keys should be a super set of the keys to select. Found input_keys "
                f"{input_keys} and keys to select {keys}."
            )
        return keys
