from math import prod

from torch import Tensor

from torchjd.aggregation._weighting_bases import GeneralizedWeighting, PSDMatrix, Weighting
from torchjd.autogram._gramian_utils import reshape_gramian


class Flattening(GeneralizedWeighting):
    """
    :class:`~torchjd.aggregation._weighting_bases.GeneralizedWeighting` flattening the generalized
    Gramian into a square matrix, extracting a vector of weights from it using a
    :class:`~torchjd.aggregation._weighting_bases.Weighting`, and returning the reshaped tensor of
    weights.

    For instance, when applied to a generalized Gramian of shape ``[2, 3, 3, 2]``, it would flatten
    it into a square Gramian matrix of shape ``[6, 6]``, apply the weighting on it to get a vector
    of weights of shape ``[6]``, and then return this vector reshaped into a matrix of shape
    ``[2, 3]``.

    :param weighting: The weighting to apply to the Gramian matrix.
    """

    def __init__(self, weighting: Weighting[PSDMatrix]):
        super().__init__()
        self.weighting = weighting

    def forward(self, generalized_gramian: Tensor) -> Tensor:
        k = generalized_gramian.ndim // 2
        shape = generalized_gramian.shape[:k]
        m = prod(shape)
        square_gramian = reshape_gramian(generalized_gramian, [m])
        weights_vector = self.weighting(square_gramian)
        weights = weights_vector.reshape(shape)
        return weights
