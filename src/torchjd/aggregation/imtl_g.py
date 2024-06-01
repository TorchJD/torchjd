import torch
from torch import Tensor
from torch.linalg import LinAlgError

from torchjd.aggregation.bases import Weighting


class IMTLGWeighting(Weighting):
    r"""
    :class:`~torchjd.aggregation.bases.Weighting` that extracts weights using a method which is
    a generalization of the method described in `Towards Impartial Multi-task Learning
    <https://discovery.ucl.ac.uk/id/eprint/10120667/>`_, supporting non-linearly independent rows
    of the matrix.

    .. admonition::
        Example

        Use IMTL-G to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import WeightedAggregator, IMTLGWeighting
        >>>
        >>> W = IMTLGWeighting()
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.0767, 1.0000, 1.0000])

        We can also call the weighting directly to get the weights vector associated to the matrix:

        >>> W(J)
        tensor([0.5923, 0.4077])
    """

    def forward(self, matrix: Tensor) -> Tensor:
        d = torch.linalg.norm(matrix, axis=1)

        try:
            raw_weights = torch.linalg.pinv(matrix @ matrix.T) @ d
        except LinAlgError:  # This can happen when the matrix has extremely large values
            raw_weights = torch.ones(matrix.shape[0])

        weights = raw_weights / raw_weights.sum()
        return weights
