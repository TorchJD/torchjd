from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence

from torch import Tensor

from .tensor_dict import _A, _B, _C, EmptyTensorDict, _least_common_ancestor


class RequirementError(ValueError):
    """Inappropriate set of inputs keys."""

    pass


class Transform(Generic[_B, _C], ABC):
    r"""
    Abstract base class for all transforms. Transforms are elementary building blocks of a jacobian
    descent backward phase. A transform maps a :class:`~torchjd.transform.tensor_dict.TensorDict` to
    another. The input :class:`~torchjd.transform.tensor_dict.TensorDict` has keys `required_keys`
    and the output :class:`~torchjd.transform.tensor_dict.TensorDict` has keys `output_keys`.

    Formally a transform is a function:

    .. math::
        f:\mathbb R^{n_1+\dots+n_p}\to \mathbb R^{m_1+\dots+m_q}

    where we have ``p`` `required_keys`, ``q`` `output_keys`, ``n_i`` is the number of elements in
    the value associated to the ``i`` th `required_key` of the input
    :class:`~torchjd.transform.tensor_dict.TensorDict` and ``m_j`` is the number of elements in the
    value associated to the ``j`` th `output_key` of the output
    :class:`~torchjd.transform.tensor_dict.TensorDict`.

    As they are mathematical functions, transforms can be composed together as long as their
    domains and range meaningfully match.
    """

    def compose(self, other: Transform[_A, _B]) -> Transform[_A, _C]:
        return Composition(self, other)

    def conjunct(self, other: Transform[_B, _C]) -> Transform[_B, _C]:
        return Conjunction([self, other])

    def __str__(self) -> str:
        return type(self).__name__

    @abstractmethod
    def __call__(self, input: _B) -> _C:
        """Applies the transform to the input."""

    @abstractmethod
    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        """
        Checks that the provided input_keys satisfy the transform's requirements and returns the
        corresponding output keys for recursion.

        If the provided input_keys do not satisfy the transform's requirements, raises a
        RequirementError.

        The output keys are the set of keys of the output TensorDict of the transform when the input
        TensorDict's keys are input_keys.
        """

    __lshift__ = compose
    __or__ = conjunct


class Composition(Transform[_A, _C]):
    def __init__(self, outer: Transform[_B, _C], inner: Transform[_A, _B]):
        self.outer = outer
        self.inner = inner

    def __str__(self) -> str:
        return str(self.outer) + " ∘ " + str(self.inner)

    def __call__(self, input: _A) -> _C:
        intermediate = self.inner(input)
        return self.outer(intermediate)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        intermediate_keys = self.inner.check_keys(input_keys)
        output_keys = self.outer.check_keys(intermediate_keys)
        return output_keys


class Conjunction(Transform[_A, _B]):
    def __init__(self, transforms: Sequence[Transform[_A, _B]]):
        self.transforms = transforms

    def __str__(self) -> str:
        strings = []
        for t in self.transforms:
            s = str(t)
            if isinstance(t, Conjunction):
                strings.append(s[1:-1])  # Remove parentheses
            else:
                strings.append(s)
        return "(" + " | ".join(strings) + ")"

    def __call__(self, tensor_dict: _A) -> _B:
        tensor_dicts = [transform(tensor_dict) for transform in self.transforms]
        output_type: type[_A] = EmptyTensorDict
        output: _A = EmptyTensorDict()
        for tensor_dict in tensor_dicts:
            output_type = _least_common_ancestor(output_type, type(tensor_dict))
            output |= tensor_dict
        return output_type(output)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        output_keys_list = [key for t in self.transforms for key in t.check_keys(input_keys)]
        output_keys = set(output_keys_list)

        if len(output_keys) != len(output_keys_list):
            raise RequirementError("The sets of output keys of transforms should be disjoint.")

        return output_keys
