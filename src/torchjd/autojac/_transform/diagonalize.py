from typing import Iterable

import torch
from torch import Tensor

from .base import RequirementError, Transform
from .ordered_set import OrderedSet
from .tensor_dict import Gradients, Jacobians


class Diagonalize(Transform[Gradients, Jacobians]):
    def __init__(self, considered: Iterable[Tensor]):
        self.considered = OrderedSet(considered)
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

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        considered = set(self.considered)
        if not considered == input_keys:
            raise RequirementError(
                f"The input_keys must match the considered keys. Found input_keys {input_keys} and"
                f"considered keys {considered}."
            )
        return considered
