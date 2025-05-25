from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Annotated, Generic, TypeVar

from torch import Tensor, nn

_T = TypeVar("_T", contravariant=True)
_FnInputT = TypeVar("_FnInputT")
_FnOutputT = TypeVar("_FnOutputT")
Matrix = Annotated[Tensor, "ndim=2"]
PSDMatrix = Annotated[Matrix, "Positive semi-definite"]


class Weighting(Generic[_T], nn.Module, ABC):
    r"""
    Abstract base class for all weighting methods. It has the role of extracting a vector of weights
    of dimension :math:`m` from some statistic of a matrix of dimension :math:`m \times n`.
    """

    @abstractmethod
    def forward(self, stat: _T) -> Tensor:
        """Computes the vector of weights from the input stat."""

    # Override to make type hints and documentation more specific
    def __call__(self, stat: _T) -> Tensor:
        """Computes the vector of weights from the input stat and applies all registered hooks."""

        return super().__call__(stat)

    def _compose(self, fn: Callable[[_FnInputT], _T]) -> Weighting[_FnInputT]:
        return _Composition(self, fn)

    __lshift__ = _compose


class _Composition(Weighting[_T]):
    """
    Weighting that composes a Weighting with a function, so that the Weighting is applied to the
    output of the function.
    """

    def __init__(self, weighting: Weighting[_FnOutputT], fn: Callable[[_T], _FnOutputT]):
        super().__init__()
        self.fn = fn
        self.weighting = weighting

    def forward(self, stat: _T) -> Tensor:
        return self.weighting(self.fn(stat))
