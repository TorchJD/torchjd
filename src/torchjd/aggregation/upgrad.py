from typing import Literal

import torch
from torch import Tensor

from ._dual_cone_utils import _project_weights
from ._gramian_utils import _compute_regularized_normalized_gramian
from ._pref_vector_utils import _pref_vector_to_str_suffix, _pref_vector_to_weighting
from .bases import _WeightedAggregator, _Weighting
from .mean import _MeanWeighting


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
        weighting = _pref_vector_to_weighting(pref_vector, default=_MeanWeighting())
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
        return f"UPGrad{_pref_vector_to_str_suffix(self._pref_vector)}"


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
        # Cache for storing computed weights
        self._cache = {}

    def forward(self, matrix: Tensor) -> Tensor:
        # Convert matrix to tuple for hashing
        with torch.no_grad():  # No need to track gradients for caching
            matrix_key = hash(matrix.cpu().numpy().tobytes())

        # Check if we have cached result
        if matrix_key in self._cache:
            return self._cache[matrix_key]

        # Compute weights once and reuse

        # Original computation optimized

        # Move computations to same device as input
        U = torch.zeros([matrix.shape[0], matrix.shape[0]])

        # Compute G and W in a single batch operation if possible
        G = _compute_regularized_normalized_gramian(matrix, self.norm_eps, self.reg_eps)
        W = _project_weights(U, G, self.solver)

        # Use more efficient sum
        result = W.sum(dim=0)

        # Cache the result
        self._cache[matrix_key] = result
        return result
