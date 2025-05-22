from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Callable, Generic, TypeVar

from torch import Tensor, nn

_A = TypeVar("_A", contravariant=True)
_B = TypeVar("_B")


class Weighting(Generic[_A], nn.Module, ABC):
    @abstractmethod
    def forward(self, stat: _A) -> Tensor:
        """Computes the vector of weights from the input stat."""

    # Override to make type hints and documentation more specific
    def __call__(self, stat: _A) -> Tensor:
        """Computes the vector of weights from the input stat and applies all registered hooks."""

        return super().__call__(stat)

    def compose(self, fn: Callable[[_B], _A]) -> Weighting[_B]:
        return Composition(self, fn)

    __lshift__ = compose


Matrix = Annotated[Tensor, "ndim=2"]
PSDMatrix = Annotated[Matrix, "Positive semi-definite"]


class Composition(Generic[_A, _B], Weighting[_A]):
    def __init__(self, weighting: Weighting[_B], fn: Callable[[_A], _B]):
        super().__init__()
        self.fn = fn
        self.weighting = weighting

    def forward(self, stat: _A) -> Tensor:
        """Computes the vector of weights from the input stat."""

        return self.weighting(self.fn(stat))
