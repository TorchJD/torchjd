from typing import Literal

import torch
from torch import Tensor

from ._dual_cone_utils import _project_weights
from ._gramian_utils import _compute_regularized_gramian
from ._pref_vector_utils import (
    _check_pref_vector,
    _pref_vector_to_str_suffix,
    _pref_vector_to_weighting,
)
from .bases import _WeightedAggregator, _Weighting
from .mean import _MeanWeighting


class UPGrad(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that projects each row of the input matrix onto
    the dual cone of all rows of this matrix, and that combines the result, as proposed in
    `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param pref_vector: The preference vector used to combine the projected rows. If not provided,
        defaults to the simple averaging of the projected rows.
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
        reg_eps: float = 0.0001,
        solver: Literal["quadprog"] = "quadprog",
    ):
        _check_pref_vector(pref_vector)
        weighting = _pref_vector_to_weighting(pref_vector, default=_MeanWeighting())
        self._pref_vector = pref_vector

        super().__init__(
            weighting=_UPGradWrapper(weighting=weighting, reg_eps=reg_eps, solver=solver)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, "
            f"reg_eps={self.weighting.reg_eps}, solver={repr(self.weighting.solver)})"
        )

    def __str__(self) -> str:
        return f"UPGrad{_pref_vector_to_str_suffix(self._pref_vector)}"


class _UPGradWrapper(_Weighting):
    """
    Wrapper of :class:`~torchjd.aggregation.bases._Weighting` that changes the weights vector such
    that each weighted row is projected onto the dual cone of all rows.

    :param weighting: The wrapped weighting.
    :param reg_eps: A small value to add to the diagonal of the gramian of the matrix. Due to
        numerical errors when computing the gramian, it might not exactly be positive definite.
        This issue can make the optimization fail. Adding ``reg_eps`` to the diagonal of the gramian
        ensures that it is positive definite.
    :param solver: The solver used to optimize the underlying optimization problem.
    """

    def __init__(
        self,
        weighting: _Weighting,
        reg_eps: float,
        solver: Literal["quadprog"],
    ):
        super().__init__()
        self.weighting = weighting
        self.reg_eps = reg_eps
        self.solver = solver

    def forward(self, matrix: Tensor) -> Tensor:
        U = torch.diag(self.weighting(matrix))
        G = _compute_regularized_gramian(matrix, self.reg_eps)
        W = _project_weights(U, G, self.solver)
        return torch.sum(W, dim=0)
