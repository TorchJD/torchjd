# The code of this file was partly adapted from https://github.com/AvivNavon/nash-mtl/tree/main.
# It is therefore also subject to the following license.
#
# MIT License
#
# Copyright (c) 2022 Aviv Navon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import cvxpy as cp
import numpy as np
import torch
from cvxpy import Expression
from torch import Tensor

from .bases import _WeightedAggregator, _Weighting


class NashMTL(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` as proposed in Algorithm 1 of
    `Multi-Task Learning as a Bargaining Game <https://arxiv.org/pdf/2202.01017.pdf>`_.

    :param n_tasks: The number of tasks, corresponding to the number of rows in the provided
        matrices.
    :param max_norm: Maximum value of the norm of :math:`A^T w`.
    :param update_weights_every: A parameter determining how often the actual weighting should be
        performed. A larger value means that the same weights will be re-used for more calls to the
        aggregator.
    :param optim_niter: The number of iterations of the underlying optimization process.

    .. admonition::
        Example

        Use NashMTL to aggregate a matrix.

        >>> import warnings
        >>> warnings.filterwarnings("ignore")
        >>>
        >>> from torch import tensor
        >>> from torchjd.aggregation import NashMTL
        >>>
        >>> A = NashMTL(n_tasks=2)
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.0542, 0.7061, 0.7061])

    .. warning::
        This implementation was adapted from the `official implementation
        <https://github.com/AvivNavon/nash-mtl/tree/main>`_, which has some flaws. Use with caution.

    .. warning::
        The aggregator is stateful. Its output will thus depend not only on the input matrix, but
        also on its state. It thus depends on previously seen matrices. It should be reset between
        experiments.
    """

    def __init__(
        self,
        n_tasks: int,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter: int = 20,
    ):
        super().__init__(
            weighting=_NashMTLWeighting(
                n_tasks=n_tasks,
                max_norm=max_norm,
                update_weights_every=update_weights_every,
                optim_niter=optim_niter,
            )
        )

    def reset(self) -> None:
        """Resets the internal state of the algorithm."""
        self.weighting.reset()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_tasks={self.weighting.n_tasks})"


class _NashMTLWeighting(_Weighting):
    """
    :class:`~torchjd.aggregation.bases._Weighting` that extracts weights using the
    step decision of Algorithm 1 of `Multi-Task Learning as a Bargaining Game
    <https://arxiv.org/pdf/2202.01017.pdf>`_.

    :param n_tasks: The number of tasks, corresponding to the number of rows in the provided
        matrices.
    :param max_norm: Maximum value of the norm of :math:`A^T w`.
    :param update_weights_every: A parameter determining how often the actual weighting should be
        performed. A larger value means that the same weights will be re-used for more calls to the
        weighting.
    :param optim_niter: The number of iterations of the underlying optimization process.
    """

    def __init__(
        self,
        n_tasks: int,
        max_norm: float,
        update_weights_every: int,
        optim_niter: int,
    ):
        super().__init__()

        self.n_tasks = n_tasks
        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def _stop_criteria(self, gtg: np.ndarray, alpha_t: np.ndarray) -> bool:
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value) < 1e-6)
        )

    def _solve_optimization(self, gtg: np.ndarray) -> np.ndarray:
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except Exception:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self) -> Expression:
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self) -> None:
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.n_tasks,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.n_tasks, self.n_tasks), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=np.array([1.0]))

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param) - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param)
        self.prob = cp.Problem(obj, constraint)

    def forward(self, matrix: Tensor) -> Tensor:
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            G = matrix
            GTG = torch.mm(G, G.t())

            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG = GTG / self.normalization_factor.item()
            alpha = self._solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha).to(device=matrix.device, dtype=matrix.dtype)
        else:
            self.step += 1
            alpha = self.prvs_alpha

        if self.max_norm > 0:
            norm = torch.linalg.norm(alpha @ matrix)
            if norm > self.max_norm:
                alpha = (alpha / norm) * self.max_norm

        return alpha

    def reset(self) -> None:
        """Resets the internal state of the algorithm."""

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)
