import torch
from torch import Tensor
from torch.linalg import LinAlgError

from torchjd.aggregation.bases import _WeightedAggregator, _Weighting


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
            raw_weights = torch.linalg.pinv(matrix @ matrix.T) @ d
        except LinAlgError:  # This can happen when the matrix has extremely large values
            raw_weights = torch.ones(matrix.shape[0])

        weights = raw_weights / raw_weights.sum()
        return weights
