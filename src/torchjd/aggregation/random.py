import torch
from torch import Tensor
from torch.nn import functional as F

from torchjd.aggregation.bases import Weighting


class RandomWeighting(Weighting):
    """
    :class:`~torchjd.aggregation.bases.Weighting` that generates positive random weights
    at each call, as defined in algorithm 2 of `Reasonable Effectiveness of Random Weighting: A
    Litmus Test for Multi-Task Learning <https://arxiv.org/pdf/2111.10603.pdf>`_.

    .. admonition::
        Example

        Compute several random combinations of the rows of a matrix.

        >>> from torch import tensor, manual_seed
        >>> from torchjd.aggregation import WeightedAggregator, RandomWeighting
        >>>
        >>> _ = torch.manual_seed(0)
        >>> W = RandomWeighting()
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([-2.6229,  1.0000,  1.0000])
        >>>
        >>> A(J)
        tensor([5.3976, 1.0000, 1.0000])

    .. admonition::
        Example

        Generate random weights for the rows of a matrix.

        >>> from torch import tensor, manual_seed
        >>> from torchjd.aggregation import WeightedAggregator, RandomWeighting
        >>>
        >>> _ = torch.manual_seed(0)
        >>> W = RandomWeighting()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> W(J)
        tensor([0.8623, 0.1377])
    """

    def forward(self, matrix: Tensor) -> Tensor:
        random_vector = torch.randn(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        weights = F.softmax(random_vector, dim=-1)
        return weights
