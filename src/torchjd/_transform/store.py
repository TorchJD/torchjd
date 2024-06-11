from typing import Iterable

from torch import Tensor

from torchjd._transform.base import Transform
from torchjd._transform.tensor_dict import EmptyTensorDict, Gradients


class Store(Transform[Gradients, EmptyTensorDict]):
    def __init__(self, required_keys: Iterable[Tensor]):
        self._required_keys = set(required_keys)

    def _compute(self, gradients: Gradients) -> EmptyTensorDict:
        """
        Stores gradients with respect to keys in their ``.grad`` field.
        """

        for key in gradients.keys():
            key.grad = gradients[key]

        return EmptyTensorDict()

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return set()
