import torch
from torch import Tensor

from .base import RequirementError, Transform
from .ordered_set import OrderedSet
from .tensor_dict import Gradients, Jacobians


class Diagonalize(Transform[Gradients, Jacobians]):
    def __init__(self, key_order: OrderedSet[Tensor]):
        self.key_order = key_order
        self.indices: list[tuple[int, int]] = []
        begin = 0
        for tensor in self.key_order:
            end = begin + tensor.numel()
            self.indices.append((begin, end))
            begin = end

    def __call__(self, tensors: Gradients) -> Jacobians:
        flattened_considered_values = [tensors[key].reshape([-1]) for key in self.key_order]
        diagonal_matrix = torch.cat(flattened_considered_values).diag()
        diagonalized_tensors = {
            key: diagonal_matrix[:, begin:end].reshape((-1,) + key.shape)
            for (begin, end), key in zip(self.indices, self.key_order)
        }
        return Jacobians(diagonalized_tensors)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        if not set(self.key_order) == input_keys:
            raise RequirementError(
                f"The input_keys must match the key_order. Found input_keys {input_keys} and"
                f"key_order {self.key_order}."
            )
        return input_keys
