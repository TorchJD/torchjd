import torch
from torch import Tensor

from .bases import _WeightedAggregator, _Weighting


class IMTLG(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` generalizing the method described in
    `Towards Impartial Multi-task Learning <https://discovery.ucl.ac.uk/id/eprint/10120667/>`_.
    This generalization supports matrices with some linearly dependant rows.

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
    :class:`~torchjd.aggregation.bases._Weighting` that extracts weights using a method which is
    a generalization of the method described in `Towards Impartial Multi-task Learning
    <https://discovery.ucl.ac.uk/id/eprint/10120667/>`_, supporting non-linearly independent rows
    of the matrix.
    """

    def forward(self, matrix: Tensor) -> Tensor:
        d = torch.linalg.norm(matrix, dim=1)

        try:
            # Equivalent to `alpha_star = torch.linalg.pinv(matrix @ matrix.T) @ d`, but safer
            # according to https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
            alpha_star = torch.linalg.lstsq(matrix @ matrix.T, d).solution
        except RuntimeError:  # This can happen when the matrix has extremely large values
            alpha_star = torch.ones(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)

        weights = alpha_star / alpha_star.sum()
        return weights
