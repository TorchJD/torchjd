from torch import Tensor

from torchjd._transform._utils import _A
from torchjd._transform.base import Transform


class Scaling(Transform[_A, _A]):
    def __init__(self, scalings: dict[Tensor, float]):
        self.scalings = scalings

    def _compute(self, tensor_dict: _A) -> _A:
        output = {key: scaling * tensor_dict[key] for key, scaling in self.scalings.items()}
        return type(tensor_dict)(output)

    @property
    def required_keys(self) -> set[Tensor]:
        return set(self.scalings.keys())

    @property
    def output_keys(self) -> set[Tensor]:
        return set(self.scalings.keys())
