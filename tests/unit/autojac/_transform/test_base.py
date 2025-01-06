import typing

import torch
from pytest import raises
from torch import Tensor

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
        # Ignore the input, create a dictionary with the right keys as an output.
        # Cast the type for the purpose of type-checking.
        output_dict = {key: torch.empty(0) for key in self._output_keys}
        return typing.cast(_C, output_dict)

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._output_keys


def test_call_checks_keys():
    """
    Tests that a ``Transform`` checks that the provided dictionary to the `__call__` function
    contains keys that correspond exactly to `required_keys`.
    """

    a1 = torch.randn([2])
    a2 = torch.randn([3])
    t = FakeTransform(required_keys={a1}, output_keys={a1, a2})

    t(TensorDict({a1: a2}))

    with raises(ValueError):
        t(TensorDict({a2: a1}))

    with raises(ValueError):
        t(TensorDict({}))

    with raises(ValueError):
        t(TensorDict({a1: a2, a2: a1}))


def test_compose_checks_keys():
    """
    Tests that the composition of ``Transform``s checks that the inner transform's `output_keys`
    match with the outer transform's `required_keys`.
    """

    a1 = torch.randn([2])
    a2 = torch.randn([3])
    t1 = FakeTransform(required_keys={a1}, output_keys={a1, a2})
    t2 = FakeTransform(required_keys={a2}, output_keys={a1})

    t1 << t2

    with raises(ValueError):
        t2 << t1


def test_conjunct_checks_required_keys():
    """
    Tests that the conjunction of ``Transform``s checks that the provided transforms all have the
    same `required_keys`.
    """

    a1 = torch.randn([2])
    a2 = torch.randn([3])

    t1 = FakeTransform(required_keys={a1}, output_keys=set())
    t2 = FakeTransform(required_keys={a1}, output_keys=set())
    t3 = FakeTransform(required_keys={a2}, output_keys=set())

    t1 | t2

    with raises(ValueError):
        t2 | t3

    with raises(ValueError):
        t1 | t2 | t3


def test_conjunct_checks_output_keys():
    """
    Tests that the conjunction of ``Transform``s checks that the transforms `output_keys` are
    disjoint.
    """

    a1 = torch.randn([2])
    a2 = torch.randn([3])

    t1 = FakeTransform(required_keys=set(), output_keys={a1, a2})
    t2 = FakeTransform(required_keys=set(), output_keys={a1})
    t3 = FakeTransform(required_keys=set(), output_keys={a2})

    t2 | t3

    with raises(ValueError):
        t1 | t3

    with raises(ValueError):
        t1 | t2 | t3


def test_empty_conjunction():
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

    t = FakeTransform(required_keys=set(), output_keys=set())
    transform = (t | t << t << t | t) << t << (t | t)

    assert str(transform) == "(T | T ∘ T ∘ T | T) ∘ T ∘ (T | T)"
