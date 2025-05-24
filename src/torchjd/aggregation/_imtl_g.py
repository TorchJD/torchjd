import torch
from torch import Tensor

from ._aggregator_bases import GramianWeightedAggregator
from ._utils.non_differentiable import raise_non_differentiable_error
from ._weighting_bases import PSDMatrix, Weighting


class IMTLG(GramianWeightedAggregator):
    """
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` generalizing the method described in
    `Towards Impartial Multi-task Learning <https://discovery.ucl.ac.uk/id/eprint/10120667/>`_.
    This generalization, defined formally in `Jacobian Descent For Multi-Objective Optimization
    <https://arxiv.org/pdf/2406.16232>`_, supports matrices with some linearly dependant rows.

    .. admonition::
        Example

        Use IMTL-G to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import IMTLG
        >>>
        >>> A = IMTLG()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.0767, 1.0000, 1.0000])
    """

    def __init__(self):
        super().__init__(_IMTLGWeighting())

        # This prevents computing gradients that can be very wrong.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)


class _IMTLGWeighting(Weighting[PSDMatrix]):
    """
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that extracts weights as described in
    the definition of A_IMTLG of `Jacobian Descent For Multi-Objective Optimization
    <https://arxiv.org/pdf/2406.16232>`_.
    """

    def forward(self, gramian: Tensor) -> Tensor:
        d = torch.sqrt(torch.diagonal(gramian))
        v = torch.linalg.pinv(gramian) @ d
        v_sum = v.sum()

        if v_sum.abs() < 1e-12:
            weights = torch.zeros_like(v)
        else:
            weights = v / v_sum

        return weights
