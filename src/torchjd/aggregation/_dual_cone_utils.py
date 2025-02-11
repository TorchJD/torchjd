from typing import Literal

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor


def _project_weights(
    U: Tensor, matrix: Tensor, solver: Literal["quadprog"], norm_eps: float, reg_eps: float
) -> Tensor:
    """
    Computes the tensor of weights corresponding to the projection of the vectors in `U` onto the
    rows of a matrix whose Gramian is provided.

    :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`.
    :param solver: The quadratic programming solver to use.
    :return: A tensor of projection weights with the same shape as `U`.
    """

    _, R = torch.linalg.qr(matrix.T, mode="r")

    # Computes the inverse of R
    dimension = matrix.shape[0]
    R_inv = torch.linalg.solve_triangular(R, torch.eye(dimension), upper=True)

    R_inv = _to_array(R_inv)
    U_ = _to_array(U)

    W = np.apply_along_axis(lambda u: _project_weight_vector(u, R_inv, solver), axis=-1, arr=U_)

    return torch.as_tensor(W, device=matrix.device, dtype=matrix.dtype)


def _project_weight_vector(
    u: np.ndarray, R_inv: np.ndarray, solver: Literal["quadprog"]
) -> np.ndarray:
    r"""
    Computes the weights `w` of the projection of `J^T u` onto the dual cone of the rows of `J`,
    given `G = J J^T` and `u`. In other words, this computes the `w` that satisfies
    `\pi_J(J^T u) = J^T w`.

    By Proposition 1 of [1], this is equivalent to solving for `v` the following quadratic program:
    minimize        v^T G v
    subject to      u \preceq v

    Reference:
    [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param u: The vector of weights `u` of shape `[m]` corresponding to the vector `J^T u` to
        project.
    :param G: The Gramian matrix of `J`, equal to `J J^T`, and of shape `[m, m]`.
    :param solver: The quadratic programming solver to use.
    """

    m = R_inv.shape[0]
    w = solve_qp(R_inv, np.zeros(m), -np.eye(m), -u, solver=solver, factorized=True)
    return w


def _to_array(tensor: Tensor) -> np.ndarray:
    """Transforms a tensor into a numpy array with float64 dtype."""

    return tensor.cpu().detach().numpy().astype(np.float64)
