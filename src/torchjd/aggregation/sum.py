import torch
from torch import Tensor

from .aggregator_bases import _WeightedAggregator
from .weighting_bases import _RowDimensionBasedWeighting


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


class _SumWeighting(_RowDimensionBasedWeighting):
    r"""
    :class:`~torchjd.aggregation.bases._Weighting` that gives the weights
    :math:`\begin{bmatrix} 1 & \dots & 1 \end{bmatrix}^T \in \mathbb{R}^m`.
    """

    def weights_from_dimension(self, m: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        weights = torch.ones(m, device=device, dtype=dtype)
        return weights
