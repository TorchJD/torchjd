from typing import Literal

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

from ._gramian_utils import _compute_normalized_gramian
from ._pref_vector_utils import _check_pref_vector, _pref_vector_to_weighting
from ._str_utils import _vector_to_str
from .bases import _WeightedAggregator, _Weighting


class DualProj(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that averages the rows of the input matrix, and
    projects the result onto the dual cone of the rows of the matrix. This corresponds to the
    solution to Equation 11 of `Gradient Episodic Memory for Continual Learning
    <https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf>`_.

    :param pref_vector: The preference vector used to combine the rows. If not provided, defaults to
        the simple averaging.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem.

    .. admonition::
        Example

        Use DualProj to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import DualProj
        >>>
        >>> A = DualProj()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.5563, 1.1109, 1.1109])
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
            weighting=_DualProjWrapper(
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
        return f"DualProj{suffix}"


class _DualProjWrapper(_Weighting):
    """
    Wrapper of :class:`~torchjd.aggregation.bases._Weighting` that changes the extracted
    weight vector such the corresponding aggregation is projected onto the dual cone of the rows
    of the input matrix. This corresponds to the solution to Equation 11 of `Gradient Episodic
    Memory for Continual Learning
    <https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf>`_.

    :param weighting: The wrapped :class:`~torchjd.aggregation.bases._Weighting`
        responsible for extracting weight vectors from the input matrices.
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
