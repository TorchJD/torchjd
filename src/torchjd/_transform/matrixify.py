from typing import Iterable

from torch import Tensor

from torchjd._transform.base import Transform
from torchjd._transform.tensor_dict import JacobianMatrices, Jacobians


class Matrixify(Transform[Jacobians, JacobianMatrices]):
    def __init__(self, required_keys: Iterable[Tensor]):
        self._required_keys = set(required_keys)

    def _compute(self, jacobians: Jacobians) -> JacobianMatrices:
        jacobian_matrices = {
            key: jacobian.view(jacobian.shape[0], -1) for key, jacobian in jacobians.items()
        }
        return JacobianMatrices(jacobian_matrices)

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._required_keys
