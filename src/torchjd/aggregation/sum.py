import torch
from torch import Tensor

from ._weighting_bases import Matrix, Weighting
from .aggregator_bases import _WeightedAggregator


class Sum(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that sums of the rows of the input matrices.

    .. admonition::
        Example

        Sum the rows of a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import Sum
        >>>
        >>> A = Sum()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([2., 2., 2.])
    """

    def __init__(self):
        super().__init__(weighting=_SumWeighting())


class _SumWeighting(Weighting[Matrix]):
    r"""
    :class:`~torchjd.aggregation.bases._RowDimBasedWeighting` that gives the weights
    :math:`\begin{bmatrix} 1 & \dots & 1 \end{bmatrix}^T \in \mathbb{R}^m`.
    """

    def forward(self, matrix: Tensor) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        weights = torch.ones(matrix.shape[0], device=device, dtype=dtype)
        return weights
