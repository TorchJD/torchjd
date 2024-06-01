from typing import Iterable

import torch
from torch import Tensor

from torchjd.transform import EmptyTensorDict, Gradients, Jacobians, Transform


class FakeGradientsTransform(Transform[EmptyTensorDict, Gradients]):
    """
    Transform that produces gradients filled with ones, for testing purposes.
    Note that it does the same thing as Init, but it does not depend on Init.
    """

    def __init__(self, keys: Iterable[Tensor]):
        self.keys = set(keys)

    def _compute(self, input: EmptyTensorDict) -> Gradients:
        return Gradients({key: torch.ones_like(key) for key in self.keys})

    @property
    def required_keys(self) -> set[Tensor]:
        return set()

    @property
    def output_keys(self) -> set[Tensor]:
        return self.keys


class FakeJacobiansTransform(Transform[EmptyTensorDict, Jacobians]):
    """
    Transform that produces jacobians filled with ones, of the specified number of rows, for testing
    purposes.
    """

    def __init__(self, keys: Iterable[Tensor], n_rows: int):
        self.keys = set(keys)
        self.n_rows = n_rows

    def _compute(self, input: EmptyTensorDict) -> Jacobians:
        return Jacobians({key: torch.ones(self.n_rows, *key.shape) for key in self.keys})

    @property
    def required_keys(self) -> set[Tensor]:
        return set()

    @property
    def output_keys(self) -> set[Tensor]:
        return self.keys
