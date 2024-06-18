from torch import Tensor
from torch.nn import functional as F

from torchjd.aggregation.bases import _Weighting


class _NormalizingWrapper(_Weighting):
    """
    Wrapper of :class:`~torchjd.aggregation.bases._Weighting` that scales the
    weights vector to have a given p-norm value.

    :param weighting: The wrapped :class:`~torchjd.aggregation.bases._Weighting`
        responsible for extracting (non-normalized) weights vectors from the input matrices.
    :param norm_p: The exponent value in the p-norm formulation for the normalization of the
        weights.
    :param norm_value: The value of the norm that we give to the weights vector.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    """

    def __init__(
        self,
        weighting: _Weighting,
        norm_p: float,
        norm_value: float,
        norm_eps: float = 1e-12,
    ):
        if norm_value < 0.0:
            raise ValueError(
                "Parameter `norm_value` should be a non-negative float. Found `norm_value = "
                f"{norm_value}`."
            )

        super().__init__()

        self.weighting = weighting
        self.norm_p = norm_p
        self.norm_value = norm_value
        self.norm_eps = norm_eps

    def forward(self, matrix: Tensor) -> Tensor:
        weights = self.weighting(matrix)
        scaled_weights = self._scale_weights(weights)
        return scaled_weights

    def _scale_weights(self, weights: Tensor) -> Tensor:
        unit_norm_weights = F.normalize(weights, p=self.norm_p, eps=self.norm_eps, dim=0)
        scaled_weights = self.norm_value * unit_norm_weights
        return scaled_weights
