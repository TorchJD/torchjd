import torch
from torch import Tensor

from .bases import _WeightedAggregator, _Weighting


class Mean(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that averages the rows of the input matrices.

    .. admonition::
        Example

        Average the rows of a matrix

        >>> from torch import tensor
        >>> from torchjd.aggregation import Mean
        >>>
        >>> A = Mean()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([1., 1., 1.])
    """

    def __init__(self):
        super().__init__(weighting=_MeanWeighting())


class _MeanWeighting(_Weighting):
    r"""
    :class:`~torchjd.aggregation.bases._Weighting` that gives the weights
    :math:`\begin{bmatrix} \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in
    \mathbb{R}^m`.
    """

    def forward(self, matrix: Tensor) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        m = matrix.shape[0]
        weights = torch.full(size=[m], fill_value=1 / m, device=device, dtype=dtype)
        return weights
