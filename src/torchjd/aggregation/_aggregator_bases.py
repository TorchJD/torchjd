from abc import ABC, abstractmethod

from torch import Tensor, nn

from ._utils.gramian import compute_gramian
from ._weighting_bases import Matrix, PSDMatrix, Weighting


class Aggregator(nn.Module, ABC):
    r"""
    Abstract base class for all aggregators. It has the role of aggregating matrices of dimension
    :math:`m \times n` into row vectors of dimension :math:`n`.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _check_is_matrix(matrix: Tensor) -> None:
        if len(matrix.shape) != 2:
            raise ValueError(
                "Parameter `matrix` should be a tensor of dimension 2. Found `matrix.shape = "
                f"{matrix.shape}`."
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


class WeightedAggregator(Aggregator):
    """
    Aggregator that combines the rows of the input jacobian matrix with weights given by applying a
    Weighting to it.

    :param weighting: The object responsible for extracting the vector of weights from the matrix.
    """

    def __init__(self, weighting: Weighting[Matrix]):
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
        weights = self.weighting(matrix)
        vector = self.combine(matrix, weights)
        return vector


class GramianWeightedAggregator(WeightedAggregator):
    """
    WeightedAggregator that computes the gramian of the input jacobian matrix before applying a
    Weighting to it.

    :param weighting: The object responsible for extracting the vector of weights from the gramian.
    """

    def __init__(self, weighting: Weighting[PSDMatrix]):
        super().__init__(weighting << compute_gramian)
