from typing import Iterable

import torch
from torch import Tensor

from ._utils import ordered_set
from .base import Transform
from .tensor_dict import Gradients, Jacobians


class Diagonalize(Transform[Gradients, Jacobians]):
    def __init__(self, considered: Iterable[Tensor]):
        self.considered = ordered_set(considered)
        self.indices: list[tuple[int, int]] = []
        begin = 0
        for tensor in self.considered:
            end = begin + tensor.numel()
            self.indices.append((begin, end))
            begin = end

    def _compute(self, tensors: Gradients) -> Jacobians:
        flattened_considered_values = [tensors[key].reshape([-1]) for key in self.considered]
        diagonal_matrix = torch.cat(flattened_considered_values).diag()
        diagonalized_tensors = {
            key: diagonal_matrix[:, begin:end].reshape((-1,) + key.shape)
            for (begin, end), key in zip(self.indices, self.considered)
        }
        return Jacobians(diagonalized_tensors)

    @property
    def required_keys(self) -> set[Tensor]:
        return set(self.considered)

    @property
    def output_keys(self) -> set[Tensor]:
        return set(self.considered)
