from typing import Iterable

import torch
from torch import Tensor

from .base import Transform
from .tensor_dict import EmptyTensorDict, Jacobians


class AccumulateGramian(Transform[Jacobians, EmptyTensorDict]):
    def __init__(self, required_keys: Iterable[Tensor], gramian: Tensor):
        self._required_keys = set(required_keys)
        self.gramian = gramian

    def _compute(self, input: Jacobians) -> EmptyTensorDict:
        for jacobian in input.values():
            self._accumulate(jacobian)
        return EmptyTensorDict()

    def _accumulate(self, jacobian: Tensor) -> None:
        if jacobian.shape[0] != self.gramian.shape[0]:
            raise ValueError("Cannot accumulate Gramians of different shapes")
        contracted_dims = list(range(1, jacobian.shape[0]))
        gramian = torch.tensordot(jacobian, jacobian, dims=(contracted_dims, contracted_dims))
        self.gramian += gramian

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return set()
