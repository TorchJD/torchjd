from collections import OrderedDict
from typing import Hashable, Iterable, Sequence, TypeAlias, TypeVar

import torch
from torch import Tensor

from .tensor_dict import EmptyTensorDict, TensorDict, _least_common_ancestor

_KeyType = TypeVar("_KeyType", bound=Hashable)
_ValueType = TypeVar("_ValueType")
_OrderedSet: TypeAlias = OrderedDict[_KeyType, None]

_A = TypeVar("_A", bound=TensorDict)
_B = TypeVar("_B", bound=TensorDict)
_C = TypeVar("_C", bound=TensorDict)


def ordered_set(elements: Iterable[_KeyType]) -> _OrderedSet[_KeyType]:
    elements = list(elements)
    result = OrderedDict.fromkeys(elements, None)
    if len(elements) != len(result):
        raise ValueError(
            f"Parameter `elements` should contain unique elements. Found `elements = {elements}`."
        )

    return result


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


def _union(tensor_dicts: Iterable[_A]) -> _A:
    output_type: type[_A] = EmptyTensorDict
    output: _A = EmptyTensorDict()
    for tensor_dict in tensor_dicts:
        output_type = _least_common_ancestor(output_type, type(tensor_dict))
        output |= tensor_dict
    return output_type(output)
