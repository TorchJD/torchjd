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

    def __call__(self, tensors: Gradients) -> Jacobians:
        flattened_considered_values = [tensors[key].reshape([-1]) for key in self.considered]
        diagonal_matrix = torch.cat(flattened_considered_values).diag()
        diagonalized_tensors = {
            key: diagonal_matrix[:, begin:end].reshape((-1,) + key.shape)
            for (begin, end), key in zip(self.indices, self.considered)
        }
        return Jacobians(diagonalized_tensors)

    def check_and_get_keys(self) -> tuple[set[Tensor], set[Tensor]]:
        keys = set(self.considered)
        return keys, keys
