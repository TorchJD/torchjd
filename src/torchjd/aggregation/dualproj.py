from typing import Literal

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

from torchjd.aggregation._utils import _compute_normalized_gramian
from torchjd.aggregation.bases import Weighting


class DualProjWrapper(Weighting):
    r"""
    Wrapper of :class:`~torchjd.aggregation.bases.Weighting` that changes the extracted
    weight vector such the corresponding aggregation is projected onto the dual cone of the rows
    of the input matrix. This corresponds to the solution to equation 11 of `Gradient Episodic
    Memory for Continual Learning
    <https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf>`_.

    :param weighting: The wrapped :class:`~torchjd.aggregation.bases.Weighting`
        responsible for extracting weight vectors from the input matrices.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem. Defaults to
        ``'quadprog'``.

    .. admonition::
        Example

        Use DualProj to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import (
        ...     WeightedAggregator,
        ...     MeanWeighting,
        ...     DualProjWrapper,
        ... )
        >>>
        >>> W = DualProjWrapper(MeanWeighting())
        >>> A = WeightedAggregator(W)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.5563, 1.1109, 1.1109])

        We can also call the weighting directly to get the weights vector associated to the matrix:

        >>> W(J)
        tensor([0.6109, 0.5000])
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
        weights_array = weights.cpu().detach().numpy()

        gramian = _compute_normalized_gramian(matrix, self.norm_eps)
        gramian_array = gramian.cpu().detach().numpy()
        dimension = gramian.shape[0]

        # Because of numerical errors, `gramian_array` might have slightly negative eigenvalue(s),
        # which makes quadprog misbehave. Adding a regularization term which is a small proportion
        # of the identity matrix ensures that the gramian is positive definite.
        regularization_array = self.reg_eps * np.eye(dimension)
        regularized_gramian_array = gramian_array + regularization_array

        P = regularized_gramian_array
        q = regularized_gramian_array @ weights_array
        G = -np.eye(dimension)
        h = np.zeros(dimension)

        projection_weights_array = solve_qp(P, q, G, h, solver=self.solver)
        projection_weights = torch.from_numpy(projection_weights_array).to(
            device=matrix.device, dtype=matrix.dtype
        )
        return projection_weights + weights

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(weighting={repr(self.weighting)}, norm_eps="
            f"{self.norm_eps}, reg_eps={self.reg_eps}, solver={repr(self.solver)})"
        )

    def __str__(self) -> str:
        return f"DualProj {self.weighting}"
