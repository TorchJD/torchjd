import torch
from torch import Tensor

from ..autogram._gramian_utils import reshape_gramian
from ._weighting_bases import GeneralizedWeighting, Weighting


class HierarchicalWeighting(GeneralizedWeighting):
    """
    Hierarchically reduces a generalized Gramian using a sequence of weighting functions.

    Applies multiple weightings in sequence to a generalized Gramian ``G`` of shape
    ``[n₁, ..., nₖ, nₖ, ..., n₁]``. It first applies the initial weighting to the innermost diagonal
    Gramians, contracts those dimensions to form a smaller generalized Gramian, and repeats the
    process with subsequent weightings. The final returned weights are chosen so that contracting
    the original Gramian directly with these weights produces the same quadratic form as applying
    the reductions step by step.

    :param weightings: A list of weighting callables, one for each hierarchical reduction step.
    """

    def __init__(self, weightings: list[Weighting]):
        super().__init__()
        self.weightings = weightings
        self.n_dims = len(weightings)

    def forward(self, generalized_gramian: Tensor) -> Tensor:

        assert len(self.weightings) * 2 == len(generalized_gramian.shape)  # temporary

        weighting = self.weightings[0]
        dim_size = generalized_gramian.shape[0]
        reshaped_gramian = reshape_gramian(generalized_gramian, [-1, dim_size])
        weights = _compute_weights(weighting, reshaped_gramian)
        generalized_gramian = _contract_gramian(reshaped_gramian, weights)

        for i in range(self.n_dim):
            weighting = self.weightings[i]
            dim_size = generalized_gramian.shape[i]
            reshaped_gramian = reshape_gramian(generalized_gramian, [-1, dim_size])
            temp_weights = _compute_weights(weighting, reshaped_gramian)
            generalized_gramian = _contract_gramian(reshaped_gramian, temp_weights)
            weights = _scale_weights(weights, temp_weights)

        return weights


def _compute_weights(weighting: Weighting, generalized_gramian: Tensor) -> Tensor:
    """
    Apply a weighting to each diagonal Gramian in a generalized Gramian.

    For a generalized Gramian ``G`` of shape ``[m, n, n, m]``, this extracts each diagonal Gramian
    ``G[j, :, :, j]`` of shape ``[n, n]`` for ``j`` in ``[m]`` and applies the provided weighting.
    The resulting weights are stacked into a tensor of shape ``[m, n]``.

    :param weighting: Callable that maps a Gramian of shape ``[n, n]`` to weights of shape ``[n]``.
    :param generalized_gramian: Tensor of shape ``[m, n, n, m]`` containing the generalized Gramian.
    :returns: Tensor of shape ``[m, n]`` containing the computed weights for each diagonal Gramian.
    """

    weights = torch.zeros(
        generalized_gramian[:2], device=generalized_gramian.device, dtype=generalized_gramian.dtype
    )
    for i in range(generalized_gramian.shape[0]):
        weights[i] = weighting(generalized_gramian[i, :, :, i])
    return weights


def _contract_gramian(generalized_gramian: Tensor, weights: Tensor) -> Tensor:
    r"""
    Compute a partial quadratic form by contracting a generalized Gramian with weight vectors on
    both sides.

    Given a generalized Gramian ``G`` of shape ``[m, n, n, m]`` and weights ``w`` of shape
    ``[m, n]``, this function computes a Gramian ``G'`` of shape ``[m, m]`` where

    .. math::

        G'[i, j] = \sum_{k, l=1}^n w[i, k] G[i, k, l, j] w[j, l].

    This can be viewed as forming a quadratic form with respect to the two innermost dimensions of
    ``G``.

    :param generalized_gramian: Tensor of shape ``[m, n, n, m]`` representing the generalized
        Gramian.
    :param weights: Tensor of shape ``[m, n]`` containing weight vectors to contract with the
        Gramian.
    :returns: Tensor of shape ``[m, m]`` containing the contracted Gramian, i.e. the partial
        quadratic form.
    """
    left_product = torch.einsum("ij,ijkl->ikl", weights, generalized_gramian)
    return torch.einsum("ij,ijl->il", weights, left_product)


def _scale_weights(weights: Tensor, scalings: Tensor) -> Tensor:
    """
    Scale a tensor along its leading dimensions by broadcasting scaling factors.

    :param weights: Tensor of shape [n₁, ..., nₖ, nₖ₊₁, ..., nₚ].
    :param scalings: Tensor of shape [n₁, ..., nₖ] providing scaling factors for the leading
        dimensions of ``weights``.
    :returns: Tensor of the same shape as ``weights``, where each slice
        ``weights[i₁, ..., iₖ, :, ..., :]`` is multiplied by ``scalings[i₁, ..., iₖ]``.
    """
    return weights * scalings[(...,) + (None,) * (weights.ndim - scalings.ndim)]
