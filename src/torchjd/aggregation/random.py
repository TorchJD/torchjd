import torch
from torch import Tensor
from torch.nn import functional as F

from .bases import _WeightedAggregator, _Weighting


class Random(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that computes a random combination of the rows of
    the provided matrices, as defined in algorithm 2 of
    `Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning
    <https://arxiv.org/pdf/2111.10603.pdf>`_.

    .. admonition::
        Example

        Compute several random combinations of the rows of a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import Random
        >>>
        >>> A = Random()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([-2.6229,  1.0000,  1.0000])
        >>>
        >>> A(J)
        tensor([5.3976, 1.0000, 1.0000])
    """

    def __init__(self):
        super().__init__(weighting=_RandomWeighting())


class _RandomWeighting(_Weighting):
    """
    :class:`~torchjd.aggregation.bases._Weighting` that generates positive random weights
    at each call, as defined in algorithm 2 of `Reasonable Effectiveness of Random Weighting: A
    Litmus Test for Multi-Task Learning <https://arxiv.org/pdf/2111.10603.pdf>`_.
    """

    def forward(self, matrix: Tensor) -> Tensor:
        random_vector = torch.randn(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        weights = F.softmax(random_vector, dim=-1)
        return weights
