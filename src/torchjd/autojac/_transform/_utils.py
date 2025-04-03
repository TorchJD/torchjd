from typing import Hashable, Iterable, Sequence, TypeVar

import torch
from torch import Tensor

_KeyType = TypeVar("_KeyType", bound=Hashable)
_ValueType = TypeVar("_ValueType")


def dicts_union(dicts: Iterable[dict[_KeyType, _ValueType]]) -> dict[_KeyType, _ValueType]:
    result = {}
    for d in dicts:
        result |= d
    return result


def _materialize(
    optional_tensors: Sequence[Tensor | None], inputs: Sequence[Tensor]
) -> tuple[Tensor, ...]:
    """
    Transforms a sequence of optional tensors by changing each None by a tensor of zeros of the same
    shape as the corresponding input. Returns the obtained sequence as a tuple.

    Note that the name "materialize" comes from the flag `materialize_grads` from
    `torch.autograd.grad`, which will be available in future torch releases.
    """

    tensors = []
    for optional_tensor, input in zip(optional_tensors, inputs):
        if optional_tensor is None:
            tensors.append(torch.zeros_like(input))
        else:
            tensors.append(optional_tensor)
    return tuple(tensors)
