from torch import Tensor

from ._aggregator_bases import WeightedAggregator
from ._utils.str import vector_to_str
from ._weighting_bases import Matrix, Weighting


class Constant(WeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that makes a linear combination of
    the rows of the provided matrix, with constant, pre-determined weights.

    :param weights: The weights associated to the rows of the input matrices.

    .. admonition::
        Example

        Compute a linear combination of the rows of a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import Constant
        >>>
        >>> A = Constant(tensor([1., 2.]))
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([8., 3., 3.])
    """

    def __init__(self, weights: Tensor):
        super().__init__(weighting=_ConstantWeighting(weights=weights))
        self._weights = weights

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weights={repr(self._weights)})"

    def __str__(self) -> str:
        weights_str = vector_to_str(self._weights)
        return f"{self.__class__.__name__}([{weights_str}])"


class _ConstantWeighting(Weighting[Matrix]):
    """
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that returns constant, pre-determined
    weights.

    :param weights: The weights associated to the rows of the input matrices.
    """

    def __init__(self, weights: Tensor):
        if weights.dim() != 1:
            raise ValueError(
                "Parameter `weights` should be a 1-dimensional tensor. Found `weights.shape = "
                f"{weights.shape}`."
            )

        super().__init__()
        self.weights = weights

    def forward(self, matrix: Tensor) -> Tensor:
        self._check_matrix_shape(matrix)
        return self.weights

    def _check_matrix_shape(self, matrix: Tensor) -> None:
        if matrix.shape[0] != len(self.weights):
            raise ValueError(
                f"Parameter `matrix` should have {len(self.weights)} rows (the number of specified "
                f"weights). Found `matrix` with {matrix.shape[0]} rows."
            )
