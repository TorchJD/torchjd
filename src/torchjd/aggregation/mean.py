import torch
from torch import Tensor

from torchjd.aggregation.bases import Weighting


class MeanWeighting(Weighting):
    r"""
    :class:`~torchjd.aggregation.bases.Weighting` that returns a vector of weights equal
    to :math:`\begin{bmatrix} \frac{1}{m} & \frac{1}{m} & \dots & \frac{1}{m} \end{bmatrix}^T \in
    \mathbb{R}^m`.

    .. admonition::
        Example

        Average the rows of a matrix

        >>> from torch import tensor
        >>> from torchjd.aggregation import WeightedAggregator, MeanWeighting
        >>>
        >>> W = MeanWeighting()
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([1., 1., 1.])

        We can also call the weighting directly to get the weights vector associated to the matrix:

        >>> W(J)
        tensor([0.5000, 0.5000])
    """

    def forward(self, matrix: Tensor) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        m = matrix.shape[0]
        weights = torch.full(size=[m], fill_value=1 / m, device=device, dtype=dtype)
        return weights
