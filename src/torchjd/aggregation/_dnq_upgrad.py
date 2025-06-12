import torch
from torch import Tensor

from ._aggregator_bases import GramianWeightedAggregator
from ._pcgrad import _PCGradWeighting
from ._utils.non_differentiable import raise_non_differentiable_error
from ._weighting_bases import PSDMatrix, Weighting


class DNQUPGrad(GramianWeightedAggregator):
    def __init__(self):
        super().__init__(weighting=_DNQUPGradWeighting())

        # This prevents considering the computed weights as constant w.r.t. the matrix.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)


class _DNQUPGradWeighting(Weighting[PSDMatrix]):
    def forward(self, gramian: Tensor) -> Tensor:
        m = gramian.shape[0]

        if m < 1:
            raise ValueError("TODO")
        elif m == 1:
            # Base case: just return a single weight of 1.
            return torch.ones((1,), device=gramian.device, dtype=gramian.dtype)
        else:
            # Divide
            sub_gramian_1 = gramian[: m // 2, : m // 2]
            sub_gramian_2 = gramian[m // 2 :, m // 2 :]

            # Conquer
            weights_1 = self(sub_gramian_1)
            weights_2 = self(sub_gramian_2)
            weights = torch.concatenate([weights_1, weights_2])

            # Recombine into 2x2 gramian
            new_gramian = self.recombine_gramian_2_2(gramian, weights)

            # TODO: reimplement this to not depend on PCGrad and to be much faster
            new_weights = _PCGradWeighting()(new_gramian)
            return torch.concatenate([weights_1 * new_weights[0], weights_2 * new_weights[1]])

    @staticmethod
    def recombine_gramian_2_2(gramian: Tensor, weights: Tensor) -> Tensor:
        weights_outer = weights.outer(weights)
        reweighted_gramian = gramian * weights_outer
        m = reweighted_gramian.shape[0]

        G00 = reweighted_gramian[: m // 2, : m // 2].sum()
        G01 = reweighted_gramian[: m // 2, m // 2 :].sum()
        G11 = reweighted_gramian[m // 2 :, m // 2 :].sum()

        return torch.tensor([[G00, G01], [G01, G11]], device=gramian.device, dtype=gramian.dtype)
