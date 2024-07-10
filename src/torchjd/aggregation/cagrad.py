import cvxpy as cp
import numpy as np
import torch
from torch import Tensor

from ._gramian_utils import _compute_normalized_gramian
from .bases import _WeightedAggregator, _Weighting


class CAGrad(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` as defined in Algorithm 1 of
    `Conflict-Averse Gradient Descent for Multi-task Learning
    <https://arxiv.org/pdf/2110.14048.pdf>`_.

    :param c: The scale of the radius of the ball constraint.
    :param norm_eps: A small value to avoid division by zero when normalizing.

    .. admonition::
        Example

        Use CAGrad to aggregate a matrix.

        >>> import warnings
        >>> warnings.filterwarnings("ignore")
        >>>
        >>> from torch import tensor
        >>> from torchjd.aggregation import CAGrad
        >>>
        >>> A = CAGrad(c=0.5)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.1835, 1.2041, 1.2041])
    """

    def __init__(self, c: float, norm_eps: float = 0.0001):
        super().__init__(weighting=_CAGradWeighting(c=c, norm_eps=norm_eps))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(c={self.weighting.c}, norm_eps={self.weighting.norm_eps})"
        )

    def __str__(self) -> str:
        c_str = str(self.weighting.c).rstrip("0")
        return f"CAGrad{c_str}"


class _CAGradWeighting(_Weighting):
    """
    :class:`~torchjd.aggregation.bases._Weighting` that extracts weights using the CAGrad
    algorithm, as defined in algorithm 1 of `Conflict-Averse Gradient Descent for Multi-task
    Learning <https://arxiv.org/pdf/2110.14048.pdf>`_.

    :param c: The scale of the radius of the ball constraint.
    :param norm_eps: A small value to avoid division by zero when normalizing.

    .. note::
        This implementation differs from the `official implementations
        <https://github.com/Cranial-XIX/CAGrad/>`_ in the way the underlying optimization problem is
        solved. This uses the `CLARABEL <https://oxfordcontrol.github.io/ClarabelDocs/stable/>`_
        solver of `cvxpy <https://www.cvxpy.org/index.html>`_ rather than the `scipy.minimize
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
        function.
    """

    def __init__(self, c: float, norm_eps: float):
        super().__init__()

        if c < 0.0:
            raise ValueError(f"Parameter `c` should be a non-negative float. Found `c = {c}`.")

        self.c = c
        self.norm_eps = norm_eps

    def forward(self, matrix: Tensor) -> Tensor:
        gramian = _compute_normalized_gramian(matrix, self.norm_eps)
        U, S, _ = torch.svd(gramian)

        reduced_matrix = U @ S.sqrt().diag()
        reduced_array = reduced_matrix.cpu().detach().numpy()

        dimension = matrix.shape[0]
        reduced_g_0 = reduced_array.T @ np.ones(dimension) / dimension
        sqrt_phi = self.c * np.linalg.norm(reduced_g_0, 2)

        w = cp.Variable(shape=dimension)
        cost = (reduced_array @ reduced_g_0).T @ w + sqrt_phi * cp.norm(reduced_array.T @ w, 2)
        problem = cp.Problem(objective=cp.Minimize(cost), constraints=[w >= 0, cp.sum(w) == 1])

        problem.solve(cp.CLARABEL)
        w_opt = w.value

        g_w_norm = np.linalg.norm(reduced_array.T @ w_opt)
        if g_w_norm >= self.norm_eps:
            weight_array = np.ones(dimension) / dimension
            weight_array += (sqrt_phi / g_w_norm) * w_opt
        else:
            # We are approximately on the pareto front
            weight_array = np.zeros(dimension)

        weights = torch.from_numpy(weight_array).to(device=matrix.device, dtype=matrix.dtype)

        return weights
