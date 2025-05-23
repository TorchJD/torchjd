from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Callable, Generic, TypeVar

from torch import Tensor, nn

_A = TypeVar("_A", contravariant=True)
_B = TypeVar("_B")
Matrix = Annotated[Tensor, "ndim=2"]
PSDMatrix = Annotated[Matrix, "Positive semi-definite"]


class Weighting(Generic[_A], nn.Module, ABC):
    r"""
    Abstract base class for all weighting methods. It has the role of extracting a vector of weights
    of dimension :math:`m` from some statistic of a matrix of dimension :math:`m \times n`.
    """

    @abstractmethod
    def forward(self, stat: _A) -> Tensor:
        """Computes the vector of weights from the input stat."""

    # Override to make type hints and documentation more specific
    def __call__(self, stat: _A) -> Tensor:
        """Computes the vector of weights from the input stat and applies all registered hooks."""

        return super().__call__(stat)

    def _compose(self, fn: Callable[[_B], _A]) -> Weighting[_B]:
        return _Composition(self, fn)

    __lshift__ = _compose


class _Composition(Generic[_A, _B], Weighting[_A]):
    """
    Weighting that composes a Weighting with a function, so that the Weighting is applied to the
    output of the function.
    """

    def __init__(self, weighting: Weighting[_B], fn: Callable[[_A], _B]):
        super().__init__()
        self.fn = fn
        self.weighting = weighting

    def forward(self, stat: _A) -> Tensor:
        return self.weighting(self.fn(stat))
