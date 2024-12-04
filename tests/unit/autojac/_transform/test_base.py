import typing

import torch
from pytest import raises
from torch import Tensor
from unit.conftest import DEVICE

from torchjd.autojac._transform._utils import _B, _C
from torchjd.autojac._transform.base import Conjunction, Transform
from torchjd.autojac._transform.tensor_dict import TensorDict


class FakeTransform(Transform[_B, _C]):
    """
    Fake ``Transform`` to test `required_keys` and `output_keys` when composing and conjuncting.
    """

    def __init__(self, required_keys: set[Tensor], output_keys: set[Tensor]):
        self._required_keys = required_keys
        self._output_keys = output_keys

    def __str__(self):
        return "T"

    def _compute(self, input: _B) -> _C:
        # ignore the input, create a dictionary with the right keys as an output.
        # cast the type for the purpose of type-checking.
        output_dict = {key: torch.empty(0, device=DEVICE) for key in self._output_keys}
        return typing.cast(_C, output_dict)

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._output_keys


def test_apply_keys():
    """
    Tests that a ``Transform`` checks that the provided dictionary to the `__apply__` function
    contains keys that correspond exactly to `required_keys`.
    """

    t1 = torch.randn([2], device=DEVICE)
    t2 = torch.randn([3], device=DEVICE)
    transform = FakeTransform({t1}, {t1, t2})

    transform(TensorDict({t1: t2}))

    with raises(ValueError):
        transform(TensorDict({t2: t1}))

    with raises(ValueError):
        transform(TensorDict({}))

    with raises(ValueError):
        transform(TensorDict({t1: t2, t2: t1}))


def test_compose_keys_match():
    """
    Tests that the composition of ``Transform``s checks that the inner transform's `output_keys`
    match with the outer transform's `required_keys`.
    """

    t1 = torch.randn([2], device=DEVICE)
    t2 = torch.randn([3], device=DEVICE)
    transform1 = FakeTransform({t1}, {t1, t2})
    transform2 = FakeTransform({t2}, {t1})

    transform1 << transform2

    with raises(ValueError):
        transform2 << transform1


def test_conjunct_required_keys():
    """
    Tests that the conjunction of ``Transform``s checks that the provided transforms all have the
    same `required_keys`.
    """

    t1 = torch.randn([2], device=DEVICE)
    t2 = torch.randn([3], device=DEVICE)

    transform1 = FakeTransform({t1}, set())
    transform2 = FakeTransform({t1}, set())
    transform3 = FakeTransform({t2}, set())

    transform1 | transform2

    with raises(ValueError):
        transform2 | transform3

    with raises(ValueError):
        transform1 | transform2 | transform3


def test_conjunct_wrong_output_keys():
    """
    Tests that the conjunction of ``Transform``s checks that the transforms `output_keys` are
    disjoint.
    """

    t1 = torch.randn([2], device=DEVICE)
    t2 = torch.randn([3], device=DEVICE)

    transform1 = FakeTransform(set(), {t1, t2})
    transform2 = FakeTransform(set(), {t1})
    transform3 = FakeTransform(set(), {t2})

    transform2 | transform3

    with raises(ValueError):
        transform1 | transform3

    with raises(ValueError):
        transform1 | transform2 | transform3


def test_conjunction_empty_transforms():
    """
    Tests that it is possible to take the conjunction of no transform. This should return an empty
    dictionary.
    """

    conjunction = Conjunction([])

    assert len(conjunction(TensorDict({}))) == 0


def test_str():
    """
    Tests that the __str__ method works correctly even for transform involving compositions and
    conjunctions.
    """

    t = FakeTransform(set(), set())
    transform = (t | t << t << t | t) << t << (t | t)

    assert str(transform) == "(T | T ∘ T ∘ T | T) ∘ T ∘ (T | T)"
