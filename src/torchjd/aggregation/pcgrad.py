import torch
from torch import Tensor

from torchjd.aggregation.bases import Weighting


class PCGradWeighting(Weighting):
    r"""
    :class:`~torchjd.aggregation.bases.Weighting` that extracts weights using the PCGrad
    algorithm, as defined in algorithm 1 of `Gradient Surgery for Multi-Task Learning
    <https://arxiv.org/pdf/2001.06782.pdf>`_.

    .. admonition::
        Example

        Use PCGrad to aggregate a matrix.

        >>> from torch import tensor, sqrt, manual_seed
        >>> from torchjd.aggregation import WeightedAggregator, PCGradWeighting
        >>>
        >>> _ = manual_seed(0)
        >>> W = PCGradWeighting()
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.5848, 3.8012, 3.8012])

        We can also call the weighting directly to get the weights vector associated to the matrix:

        >>> W(J)
        tensor([2.2222, 1.5789])

    .. note::
        This implementation corresponds to the paper's algorithm, which differs from the `official
        implementation <https://github.com/tianheyu927/PCGrad>`_ in the way randomness is handled.
    """

    def forward(self, matrix: Tensor) -> Tensor:
        # Pre-compute the inner products
        inner_products = matrix @ matrix.T

        # Move all computations on cpu to avoid moving memory between cpu and gpu at each iteration
        device = matrix.device
        dtype = matrix.dtype
        cpu = torch.device("cpu")
        inner_products = inner_products.to(device=cpu)

        dimension = inner_products.shape[0]
        weights = torch.zeros(dimension, device=cpu, dtype=dtype)

        for i in range(dimension):
            permutation = torch.randperm(dimension)
            current_weights = torch.zeros(dimension, device=cpu, dtype=dtype)
            current_weights[i] = 1.0

            for j in permutation:
                if j == i:
                    continue

                # Compute the inner product between g_i^{PC} and g_j
                inner_product = inner_products[j] @ current_weights

                if inner_product < 0.0:
                    current_weights[j] -= inner_product / (inner_products[j, j])

            weights = weights + current_weights

        return weights.to(device)
