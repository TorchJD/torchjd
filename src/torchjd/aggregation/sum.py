import torch
from torch import Tensor

from torchjd.aggregation.bases import _WeightedAggregator, _Weighting


class Sum(_WeightedAggregator):
    """
    TODO

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


class _SumWeighting(_Weighting):
    r"""
    :class:`~torchjd.aggregation.bases._Weighting` that gives the weights
    :math:`\begin{bmatrix} 1 & \dots & 1 \end{bmatrix}^T \in \mathbb{R}^m`.
    """

    def forward(self, matrix: Tensor) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        weights = torch.ones(matrix.shape[0], device=device, dtype=dtype)
        return weights
