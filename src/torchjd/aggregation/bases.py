from abc import ABC, abstractmethod

from torch import Tensor, nn


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
        raise NotImplementedError

    # Override to make type hints more specific
    def __call__(self, matrix: Tensor) -> Tensor:
        return super().__call__(matrix)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class _Weighting(nn.Module, ABC):
    r"""
    Abstract base class for all weighting methods. It has the role of extracting a vector of weights
    of dimension :math:`m` from a matrix of dimension :math:`m \times n`.
    """

    def __init__(self):
        super().__init__()

    def forward(self, matrix: Tensor) -> Tensor:
        raise NotImplementedError

    # Override to make type hints more specific
    def __call__(self, matrix: Tensor) -> Tensor:
        return super().__call__(matrix)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class _WeightedAggregator(Aggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that combines the rows of the input matrix with
    weights given by applying a :class:`~torchjd.aggregation.bases._Weighting` to the matrix.

    :param weighting: The object responsible for extracting the vector of weights from the matrix.
    """

    def __init__(self, weighting: _Weighting):
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
