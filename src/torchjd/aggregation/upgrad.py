from typing import Literal

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

from torchjd.aggregation._utils import _compute_normalized_gramian
from torchjd.aggregation.bases import Weighting


class UPGradWrapper(Weighting):
    r"""
    Wrapper of :class:`~torchjd.aggregation.bases.Weighting` that changes the weights vector such
    that each weighted row is projected onto the dual cone of all rows. If the wrapped weighting is
    :class:`~torchjd.aggregation.mean.Mean`, this corresponds exactly to UPGrad, as defined in our
    paper.

    :param weighting: The wrapped weight weighting.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem. Defaults to
        ``'quadprog'``.

    .. admonition::
        Example

        Use UPGrad to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import WeightedAggregator, UPGradWrapper, MeanWeighting
        >>>
        >>> W = UPGradWrapper(MeanWeighting())
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.2929, 1.9004, 1.9004])

        We can also call the weighting directly to get the weights vector associated to the matrix:

        >>> W(J)
        tensor([1.1109, 0.7894])
    """

    def __init__(
        self,
        weighting: Weighting,
        norm_eps: float = 0.0001,
        reg_eps: float = 0.0001,
        solver: Literal["quadprog"] = "quadprog",
    ):
        super().__init__()
        self.weighting = weighting
        self.norm_eps = norm_eps
        self.reg_eps = reg_eps
        self.solver = solver

    def forward(self, matrix: Tensor) -> Tensor:
        weights = self.weighting(matrix)
        lagrangian = self._compute_lagrangian(matrix, weights)
        lagrangian_weights = torch.sum(lagrangian, dim=0)
        result_weights = lagrangian_weights + weights
        return result_weights

    def _compute_lagrangian(self, matrix: Tensor, weights: Tensor) -> Tensor:
        gramian = _compute_normalized_gramian(matrix, self.norm_eps)
        gramian_array = gramian.cpu().detach().numpy()
        dimension = gramian.shape[0]

        regularization_array = self.reg_eps * np.eye(dimension)
        regularized_gramian_array = gramian_array + regularization_array

        P = regularized_gramian_array
        G = -np.eye(dimension)
        h = np.zeros(dimension)

        lagrangian_rows = []
        for i in range(dimension):
            weight = weights[i].item()
            if weight <= 0.0:
                # In this case, the solution to the quadratic program is always 0,
                # so we don't need to run solve_qp.
                lagrangian_rows.append(np.zeros([dimension]))
            else:
                q = weight * regularized_gramian_array[i, :]
                lagrangian_rows.append(solve_qp(P, q, G, h, solver=self.solver))

        lagrangian_array = np.stack(lagrangian_rows)
        lagrangian = torch.from_numpy(lagrangian_array).to(
            device=gramian.device, dtype=gramian.dtype
        )
        return lagrangian

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(weighting={repr(self.weighting)}, norm_eps="
            f"{self.norm_eps}, reg_eps={self.reg_eps}, solver={repr(self.solver)})"
        )

    def __str__(self) -> str:
        return f"UPGrad {self.weighting}"
