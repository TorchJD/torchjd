from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence

from torch import Tensor

from ._utils import _A, _B, _C, _union


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
    def _compute(self, input: _B) -> _C:
        raise NotImplementedError

    def __call__(self, input: _B) -> _C:
        input.check_keys_are(self.required_keys)
        return self._compute(input)

    @property
    @abstractmethod
    def required_keys(self) -> set[Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_keys(self) -> set[Tensor]:
        raise NotImplementedError

    __lshift__ = compose
    __or__ = conjunct


class Composition(Transform[_A, _C]):
    def __init__(self, outer: Transform[_B, _C], inner: Transform[_A, _B]):
        if outer.required_keys != inner.output_keys:
            raise ValueError(
                "The `output_keys` of `inner` must match with the `required_keys` of "
                f"outer. Found {outer.required_keys} and {inner.output_keys}"
            )
        self.outer = outer
        self.inner = inner

    def __str__(self) -> str:
        return str(self.outer) + " ∘ " + str(self.inner)

    def _compute(self, input: _A) -> _C:
        intermediate = self.inner(input)
        return self.outer(intermediate)

    @property
    def required_keys(self) -> set[Tensor]:
        return self.inner.required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self.outer.output_keys


class Conjunction(Transform[_A, _B]):
    def __init__(self, transforms: Sequence[Transform[_A, _B]]):
        self.transforms = transforms

        self._required_keys = set(
            key for transform in transforms for key in transform.required_keys
        )
        for transform in transforms:
            if transform.required_keys != self.required_keys:
                raise ValueError("All transforms should require the same set of keys.")

        output_keys_with_duplicates = [key for t in transforms for key in t.output_keys]
        self._output_keys = set(output_keys_with_duplicates)

        if len(self._output_keys) != len(output_keys_with_duplicates):
            raise ValueError("The sets of output keys of transforms should be disjoint.")

    def __str__(self) -> str:
        strings = []
        for t in self.transforms:
            s = str(t)
            if isinstance(t, Conjunction):
                strings.append(s[1:-1])  # Remove parentheses
            else:
                strings.append(s)
        return "(" + " | ".join(strings) + ")"

    def _compute(self, tensor_dict: _A) -> _B:
        output = _union([transform(tensor_dict) for transform in self.transforms])
        return output

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._output_keys
