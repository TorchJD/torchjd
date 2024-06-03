import typing

import pytest
import torch
from torch import Tensor

from torchjd.transform._utils import _B, _C
from torchjd.transform.base import Conjunction, Transform
from torchjd.transform.tensor_dict import TensorDict


class MockTransform(Transform[_B, _C]):
    def __init__(self, required_keys: set[Tensor], output_keys: set[Tensor]):
        self._required_keys = required_keys
        self._output_keys = output_keys

    def _compute(self, input: _B) -> _C:
        output_dict = {key: None for key in self._output_keys}
        return typing.cast(_C, output_dict)

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._output_keys


class MockTensorDict(TensorDict):
    pass


def test_apply_keys():
    t1 = torch.randn([2])
    t2 = torch.randn([3])
    transform = MockTransform({t1}, {t1, t2})

    transform(MockTensorDict({t1: t2}))

    with pytest.raises(ValueError):
        transform(MockTensorDict({t2: t1}))

    with pytest.raises(ValueError):
        transform(MockTensorDict({}))

    with pytest.raises(ValueError):
        transform(MockTensorDict({t1: t2, t2: t1}))


def test_compose_keys_match():
    t1 = torch.randn([2])
    t2 = torch.randn([3])
    transform1 = MockTransform({t1}, {t1, t2})
    transform2 = MockTransform({t2}, {t1})

    transform1 << transform2

    with pytest.raises(ValueError):
        transform2 << transform1


def test_conjunct_required_keys():
    t1 = torch.randn([2])
    t2 = torch.randn([3])

    transform1 = MockTransform({t1}, set())
    transform2 = MockTransform({t1}, set())
    transform3 = MockTransform({t2}, set())

    transform1 | transform2

    with pytest.raises(ValueError):
        transform2 | transform3

    with pytest.raises(ValueError):
        transform1 | transform2 | transform3


def test_conjunct_wrong_output_keys():
    t1 = torch.randn([2])
    t2 = torch.randn([3])

    transform1 = MockTransform(set(), {t1, t2})
    transform2 = MockTransform(set(), {t1})
    transform3 = MockTransform(set(), {t2})

    transform2 | transform3

    with pytest.raises(ValueError):
        transform1 | transform3

    with pytest.raises(ValueError):
        transform1 | transform2 | transform3


def test_conjunction_empty_transforms():
    conjunction = Conjunction([])

    assert len(conjunction(MockTensorDict({}))) == 0
