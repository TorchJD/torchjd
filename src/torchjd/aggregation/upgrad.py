from typing import Literal

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

from ._gramian_utils import _compute_normalized_gramian
from ._pref_vector_utils import _check_pref_vector, _pref_vector_to_weighting
from ._str_utils import _vector_to_str
from .bases import _WeightedAggregator, _Weighting


class UPGrad(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that projects each row of the input matrix onto
    the dual cone of all rows of this matrix, and that combines the result, as proposed in
    `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param pref_vector: The preference vector used to combine the projected rows. If not provided,
        defaults to the simple averaging of the projected rows.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem.

    .. admonition::
        Example

        Use UPGrad to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import UPGrad
        >>>
        >>> A = UPGrad()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.2929, 1.9004, 1.9004])
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        norm_eps: float = 0.0001,
        reg_eps: float = 0.0001,
        solver: Literal["quadprog"] = "quadprog",
    ):
        _check_pref_vector(pref_vector)
        weighting = _pref_vector_to_weighting(pref_vector)
        self._pref_vector = pref_vector

        super().__init__(
            weighting=_UPGradWrapper(
                weighting=weighting, norm_eps=norm_eps, reg_eps=reg_eps, solver=solver
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, norm_eps="
            f"{self.weighting.norm_eps}, reg_eps={self.weighting.reg_eps}, "
            f"solver={repr(self.weighting.solver)})"
        )

    def __str__(self) -> str:
        if self._pref_vector is None:
            suffix = ""
        else:
            suffix = f"([{_vector_to_str(self._pref_vector)}])"
        return f"UPGrad{suffix}"


class _UPGradWrapper(_Weighting):
    """
    Wrapper of :class:`~torchjd.aggregation.bases._Weighting` that changes the weights vector such
    that each weighted row is projected onto the dual cone of all rows.

    :param weighting: The wrapped weighting.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem.
    """

    def __init__(
        self,
        weighting: _Weighting,
        norm_eps: float,
        reg_eps: float,
        solver: Literal["quadprog"],
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
