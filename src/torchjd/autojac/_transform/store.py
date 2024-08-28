from typing import Iterable

from torch import Tensor

from .base import Transform
from .tensor_dict import EmptyTensorDict, Gradients


class Store(Transform[Gradients, EmptyTensorDict]):
    def __init__(self, required_keys: Iterable[Tensor]):
        self._required_keys = set(required_keys)

    def _compute(self, gradients: Gradients) -> EmptyTensorDict:
        """
        Stores gradients with respect to keys in their ``.grad`` field.
        """

        for key in gradients.keys():
            _check_expects_grad(key)
            key.grad = gradients[key]

        return EmptyTensorDict()

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return set()


def _check_expects_grad(tensor: Tensor) -> None:
    if not _expects_grad(tensor):
        raise ValueError(
            "Cannot populate the .grad field of a Tensor that does not satisfy:"
            "`tensor.requires_grad and (tensor.is_leaf or tensor.retains_grad)`."
        )


def _expects_grad(tensor: Tensor) -> bool:
    """
    Determines whether a Tensor expects its .grad attribute to be populated.
    See https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf for more information.
    """

    return tensor.requires_grad and (tensor.is_leaf or tensor.retains_grad)
