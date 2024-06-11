from typing import Iterable

from torch import Tensor

from torchjd._transform.base import Transform
from torchjd._transform.tensor_dict import Gradients, GradientVectors


class Reshape(Transform[GradientVectors, Gradients]):
    def __init__(self, required_keys: Iterable[Tensor]):
        self._required_keys = set(required_keys)

    def _compute(self, gradient_vectors: GradientVectors) -> Gradients:
        gradients = {
            key: gradient_vector.view(key.shape)
            for key, gradient_vector in gradient_vectors.items()
        }
        return Gradients(gradients)

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._required_keys
