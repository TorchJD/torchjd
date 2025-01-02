import torch
from torch import Tensor

from torchjd.aggregation.bases import _WeightedAggregator, _Weighting


class ConFIG(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` as defined in Equation 2 of `ConFIG: Towards
    Conflict-free Training of Physics Informed Neural Networks <https://arxiv.org/pdf/2408.11104>`_.

    .. admonition::
        Example

        Use ConFIG to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import ConFIG
        >>>
        >>> A = ConFIG()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        # TODO: add result
        # TODO: add doc test
    """

    def __init__(self):
        super().__init__(weighting=_ConFIGWeighting())


class _ConFIGWeighting(_Weighting):
    """
    TODO
    """

    def forward(self, matrix: Tensor) -> Tensor:
        # TODO
        return torch.ones(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
