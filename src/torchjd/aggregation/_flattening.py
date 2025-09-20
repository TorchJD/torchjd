from math import prod

from torch import Tensor

from torchjd.aggregation._weighting_bases import GeneralizedWeighting, PSDMatrix, Weighting
from torchjd.autogram._gramian_utils import reshape_gramian


class Flattening(GeneralizedWeighting):
    """
    :class:`~torchjd.aggregation._weighting_bases.GeneralizedWeighting` flattening the Gramian,
    extracting a vector of weights from it using a
    :class:`~torchjd.aggregation._weighting_bases.Weighting`, and returning the reshaped tensor of
    weights.

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
