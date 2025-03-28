import torch
from torch import Tensor

from .bases import _WeightedAggregator, _Weighting


class IMTLG(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` generalizing the method described in
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
        super().__init__(weighting=_IMTLGWeighting())


class _IMTLGWeighting(_Weighting):
    """
    :class:`~torchjd.aggregation.bases._Weighting` that extracts weights as described in the
    definition of A_IMTLG of `Jacobian Descent For Multi-Objective Optimization
    <https://arxiv.org/pdf/2406.16232>`_.
    """

    def forward(self, matrix: Tensor) -> Tensor:
        d = torch.linalg.norm(matrix, dim=1)
        v = torch.linalg.pinv(matrix @ matrix.T) @ d
        v_sum = v.sum()

        if v_sum.abs() < 1e-12:
            weights = torch.zeros_like(v)
        else:
            weights = v / v_sum

        return weights
