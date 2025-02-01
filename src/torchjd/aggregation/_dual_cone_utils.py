from typing import Literal

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor


def _get_projection_weights(
    gramian: Tensor, weights: Tensor, solver: Literal["quadprog"]
) -> Tensor:
    """
    Computes the weights of the projection of some weights onto the dual cone of a matrix whose
    gramian is provided. Specifically, this solves for $w$ in the problem defined by (5) in
    Proposition 1 of [1] when the gramian is $JJ^\top$ and $v$ is given by weights.
    This is a vectorized version, therefore weights can be a matrix made of columns of weights.

    [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.
    """
    shape = weights.shape
    if len(shape) == 1:
        weights_list = [weights.cpu().detach().numpy().astype(np.float64)]
    elif len(shape) == 2:
        weights_list = [weight.cpu().detach().numpy().astype(np.float64) for weight in weights.T]
    else:
        raise ValueError(f"Expect vector or matrix of weights, found shape {shape}")

    gramian_array = gramian.cpu().detach().numpy().astype(np.float64)

    lagrangian_rows = []
    for weight_array in weights_list:
        lagrangian_rows.append(_get_lagrange_multipliers(gramian_array, weight_array, solver))

    lagrangian_array = np.stack(lagrangian_rows).T.reshape(shape)
    lagrangian = torch.from_numpy(lagrangian_array).to(device=gramian.device, dtype=gramian.dtype)
    return lagrangian + weights


def _get_lagrange_multipliers(
    gramian_array: np.array, weight_array: np.array, solver: Literal["quadprog"]
) -> np.array:
    """
    Solves the dual of the projection of a vector of weights onto the dual cone of the matrix J
    whose gramian is given.
    """
    dimension = gramian_array.shape[0]
    P = gramian_array
    q = gramian_array @ weight_array
    G = -np.eye(dimension)
    h = np.zeros(dimension)
    return solve_qp(P, q, G, h, solver=solver)
