from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Callable, Generic, TypeVar

from torch import Tensor, nn

from ._utils.gramian import compute_gramian

_T = TypeVar("_T", contravariant=True)
_FnInputT = TypeVar("_FnInputT")
_FnOutputT = TypeVar("_FnOutputT")
Matrix = Annotated[Tensor, "ndim=2"]
PSDMatrix = Annotated[Matrix, "Positive semi-definite"]


class Aggregator(nn.Module, ABC):
    r"""
    Abstract base class for all aggregators. It has the role of aggregating matrices of dimension
    :math:`m \times n` into row vectors of dimension :math:`n`.
    """

    @staticmethod
    def _check_is_matrix(matrix: Tensor) -> None:
        if len(matrix.shape) != 2:
            raise ValueError(
                "Parameter `matrix` should be a tensor of dimension 2. Found `matrix.shape = "
                f"{matrix.shape}`."
            )

    @staticmethod
    def _check_is_finite(matrix: Tensor) -> None:
        if not matrix.isfinite().all():
            raise ValueError(
                "Parameter `matrix` should be a tensor of finite elements (no nan, inf or -inf "
                f"values). Found `matrix = {matrix}`."
            )

    @abstractmethod
    def forward(self, matrix: Tensor) -> Tensor:
        """Computes the aggregation from the input matrix."""

    # Override to make type hints and documentation more specific
    def __call__(self, matrix: Tensor) -> Tensor:
        """Computes the aggregation from the input matrix and applies all registered hooks."""

        return super().__call__(matrix)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class _WeightedAggregator(Aggregator):
    """
    Aggregator that combines the rows of the input jacobian matrix with weights given by applying a
    Weighting to it.

    :param weighting: The object responsible for extracting the vector of weights from the matrix.
    """

    def __init__(self, weighting: _Weighting[Matrix]):
        super().__init__()
        self.weighting = weighting

    @staticmethod
    def combine(matrix: Tensor, weights: Tensor) -> Tensor:
        """
        Aggregates a matrix by making a linear combination of its rows, using the provided vector of
        weights.
        """

        vector = weights @ matrix
        return vector

    def forward(self, matrix: Tensor) -> Tensor:
        self._check_is_matrix(matrix)
        self._check_is_finite(matrix)

        weights = self.weighting(matrix)
        vector = self.combine(matrix, weights)
        return vector


class _GramianWeightedAggregator(_WeightedAggregator):
    """
    WeightedAggregator that computes the gramian of the input jacobian matrix before applying a
    Weighting to it.

    :param weighting: The object responsible for extracting the vector of weights from the gramian.
    """

    def __init__(self, weighting: _Weighting[PSDMatrix]):
        super().__init__(weighting << compute_gramian)


class _Weighting(Generic[_T], nn.Module, ABC):
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

    def _compose(self, fn: Callable[[_FnInputT], _T]) -> _Weighting[_FnInputT]:
        return _Composition(self, fn)

    __lshift__ = _compose


class _Composition(_Weighting[_T]):
    """
    Weighting that composes a Weighting with a function, so that the Weighting is applied to the
    output of the function.
    """

    def __init__(self, weighting: _Weighting[_FnOutputT], fn: Callable[[_T], _FnOutputT]):
        super().__init__()
        self.fn = fn
        self.weighting = weighting

    def forward(self, stat: _T) -> Tensor:
        return self.weighting(self.fn(stat))
