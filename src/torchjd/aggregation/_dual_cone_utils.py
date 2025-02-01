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
    lagrange_multipliers = _get_lagrange_multipliers(gramian, weights, solver)
    return lagrange_multipliers + weights


def _get_lagrange_multipliers(
    gramian: Tensor, weights: Tensor, solver: Literal["quadprog"]
) -> Tensor:
    weights_array = weights.cpu().detach().numpy().astype(np.float64)
    shape = weights.shape
    if len(shape) == 1:
        weights_list = [weights_array]
    elif len(shape) == 2:
        weights_list = [weight for weight in weights_array.T]
    else:
        raise ValueError(f"Expect vector or matrix of weights, found shape {shape}")

    gramian_array = gramian.cpu().detach().numpy().astype(np.float64)

    lagrange_multipliers_rows = []
    for weight_array in weights_list:
        lagrange_multipliers_rows.append(
            _get_lagrange_multipliers_array(gramian_array, weight_array, solver)
        )

    lagrange_array = np.stack(lagrange_multipliers_rows).T.reshape(shape)
    lagrange_multipliers = torch.from_numpy(lagrange_array).to(
        device=gramian.device, dtype=gramian.dtype
    )
    return lagrange_multipliers


def _get_lagrange_multipliers_array(
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
