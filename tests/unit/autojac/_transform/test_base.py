import typing

import torch
from pytest import raises
from torch import Tensor

from torchjd.autojac._transform.base import Conjunction, RequirementError, Transform
from torchjd.autojac._transform.tensor_dict import _B, _C, TensorDict


class FakeTransform(Transform[_B, _C]):
    """
    Fake ``Transform`` to test `check_keys` when composing and conjuncting.
    """

    def __init__(self, required_keys: set[Tensor], output_keys: set[Tensor]):
        self._required_keys = required_keys
        self._output_keys = output_keys

    def __str__(self):
        return "T"

    def __call__(self, input: _B) -> _C:
        # Ignore the input, create a dictionary with the right keys as an output.
        # Cast the type for the purpose of type-checking.
        output_dict = {key: torch.empty(0) for key in self._output_keys}
        return typing.cast(_C, output_dict)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        # Arbitrary requirement for testing purposes.
        if not input_keys == self._required_keys:
            raise RequirementError()
        return self._output_keys


def test_composition_check_keys():
    """
    Tests that `check_keys` works correctly for a composition of transforms: the inner transform's
    `output_keys` has to satisfy the outer transform's requirements.
    """

    a1 = torch.randn([2])
    a2 = torch.randn([3])
    t1 = FakeTransform(required_keys={a1}, output_keys={a1, a2})
    t2 = FakeTransform(required_keys={a2}, output_keys={a1})

    output_keys = (t1 << t2).check_keys({a2})
    assert output_keys == {a1, a2}

    # Inner Transform fails its check
    with raises(RequirementError):
        (t1 << t2).check_keys({a1})

    # Outer Transform fails its check
    with raises(RequirementError):
        (t2 << t1).check_keys({a1})


def test_conjunct_check_keys_1():
    """
    Tests that `check_keys` works correctly for a conjunction of transforms: all transforms should
    successfully check their keys.
    """

    a1 = torch.randn([2])
    a2 = torch.randn([3])

    t1 = FakeTransform(required_keys={a1}, output_keys=set())
    t2 = FakeTransform(required_keys={a1}, output_keys=set())
    t3 = FakeTransform(required_keys={a2}, output_keys=set())

    output_keys = (t1 | t2).check_keys({a1})
    assert output_keys == set()

    with raises(RequirementError):
        (t2 | t3).check_keys({a1, a2})

    with raises(RequirementError):
        (t1 | t2 | t3).check_keys({a1, a2})


def test_conjunct_check_keys_2():
    """
    Tests that `check_keys` works correctly for a conjunction of transforms: their `output_keys`
    should be disjoint.
    """

    a1 = torch.randn([2])
    a2 = torch.randn([3])

    t1 = FakeTransform(required_keys=set(), output_keys={a1, a2})
    t2 = FakeTransform(required_keys=set(), output_keys={a1})
    t3 = FakeTransform(required_keys=set(), output_keys={a2})

    output_keys = (t2 | t3).check_keys(set())
    assert output_keys == {a1, a2}

    with raises(RequirementError):
        (t1 | t3).check_keys(set())

    with raises(RequirementError):
        (t1 | t2 | t3).check_keys(set())


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
