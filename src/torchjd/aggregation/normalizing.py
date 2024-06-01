from torch import Tensor
from torch.nn import functional as F

from torchjd.aggregation.bases import Weighting


class NormalizingWrapper(Weighting):
    """
    Wrapper of :class:`~torchjd.aggregation.bases.Weighting` that scales the
    weights vector to have a given p-norm value.

    :param weighting: The wrapped :class:`~torchjd.aggregation.bases.Weighting`
        responsible for extracting (non-normalized) weights vectors from the input matrices.
    :param norm_p: The exponent value in the p-norm formulation for the normalization of the
        weights.
    :param norm_value: The value of the norm that we give to the weights vector.
    :param norm_eps: A small value to avoid division by zero when normalizing.

    .. admonition::
        Example

        Compute a linear combination of the rows of a matrix, such that the absolute values of the
        weights sum to 1 (i.e. the :math:`L_1` norm of the weights vector is equal to 1).

        This example can be read in contrast with the example from
        :class:`~torchjd.aggregation.constant.ConstantWeighting` (same thing without normalizing
        the weights vector).

        >>> from torch import tensor
        >>> from torchjd.aggregation import (
        ...     WeightedAggregator,
        ...     ConstantWeighting,
        ...     NormalizingWrapper,
        ... )
        >>>
        >>> W = NormalizingWrapper(ConstantWeighting(tensor([1., 2.])), norm_p=1., norm_value=1.)
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([2.6667, 1.0000, 1.0000])

        We can also call the weighting directly to get the weights vector associated to the matrix:

        >>> W(J)
        tensor([0.3333, 0.6667])
    """

    def __init__(
        self,
        weighting: Weighting,
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(weighting={repr(self.weighting)}, norm_p="
            f"{self.norm_p}, norm_value={self.norm_value}, norm_eps={self.norm_eps})"
        )

    def __str__(self) -> str:
        return f"Norm{self.norm_p}-{self.norm_value} {self.weighting}"
