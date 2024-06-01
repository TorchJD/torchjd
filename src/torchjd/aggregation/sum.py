import torch
from torch import Tensor

from torchjd.aggregation.bases import Weighting


class SumWeighting(Weighting):
    r"""
    :class:`~torchjd.aggregation.bases.Weighting` that gives weights equal to 1.

    .. admonition::
        Example

        Sum the rows of a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import WeightedAggregator, SumWeighting
        >>>
        >>> W = SumWeighting()
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([2., 2., 2.])

        We can also call the weighting directly to get the weights vector associated to the matrix:

        >>> W(J)
        tensor([1., 1.])
    """

    def forward(self, matrix: Tensor) -> Tensor:
        device = matrix.device
        dtype = matrix.dtype
        weights = torch.ones(matrix.shape[0], device=device, dtype=dtype)
        return weights
