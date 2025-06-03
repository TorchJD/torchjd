from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic

from torch import Tensor

from ._tensor_dict import _A, _B, _C, TensorDict


class RequirementError(ValueError):
    """Inappropriate set of inputs keys."""

    pass


class Transform(Generic[_B, _C], ABC):
    """
    Abstract base class for all transforms. Transforms are elementary building blocks of a jacobian
    descent backward phase. A transform maps a TensorDict to another.
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


class Composition(Transform[_B, _C]):
    """
    Transform corresponding to the composition of two transforms inner and outer.

    :param inner: The transform to apply first, to the input.
    :param outer: The transform to apply second, to the result of ``inner``.
    """

    def __init__(self, outer: Transform[_A, _C], inner: Transform[_B, _A]):
        self.outer = outer
        self.inner = inner

    def __str__(self) -> str:
        return str(self.outer) + " ∘ " + str(self.inner)

    def __call__(self, input: _B) -> _C:
        intermediate = self.inner(input)
        return self.outer(intermediate)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        intermediate_keys = self.inner.check_keys(input_keys)
        output_keys = self.outer.check_keys(intermediate_keys)
        return output_keys


class Conjunction(Transform[_B, _C]):
    """
    Transform applying several transforms to the same input, and combining the results (by union)
    into a single TensorDict.

    :param transforms: The non-empty sequence of transforms to apply. Their outputs should have
        disjoint sets of keys.
    """

    def __init__(self, transforms: Sequence[Transform[_B, _C]]):
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

    def __call__(self, tensor_dict: _B) -> Any:
        # Technically, we should infer what _C is (it might not just be type(self.transforms[0]))
        # and instantiate _C(union), but this is too costly so we just return a TensorDict and use
        # the `Any` return type hint to indicate to the type checker to be lenient here.

        union: dict[Tensor, Tensor] = {}
        for transform in self.transforms:
            union |= transform(tensor_dict)
        return TensorDict(union)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        output_keys_list = [key for t in self.transforms for key in t.check_keys(input_keys)]
        output_keys = set(output_keys_list)

        if len(output_keys) != len(output_keys_list):
            raise RequirementError("The sets of output keys of transforms should be disjoint.")

        return output_keys
