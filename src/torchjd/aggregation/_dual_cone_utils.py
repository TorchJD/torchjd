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
    weight_matrix = _to_array(weights.reshape([-1, weights.shape[-1]]))
    gramian_array = _to_array(gramian)

    lagrange_multiplier_vectors = [
        _get_lagrange_multiplier_vector(gramian_array, weight_vector, solver)
        for weight_vector in weight_matrix
    ]

    lagrange_multiplier_matrix = np.stack(lagrange_multiplier_vectors).T
    lagrange_multipliers = (
        torch.from_numpy(lagrange_multiplier_matrix)
        .to(device=gramian.device, dtype=gramian.dtype)
        .reshape(weights.shape)
    )
    return lagrange_multipliers


def _get_lagrange_multiplier_vector(
    gramian: np.array, weight_vector: np.array, solver: Literal["quadprog"]
) -> np.array:
    """
    Solves the dual of the projection of a vector of weights onto the dual cone of the matrix J
    whose gramian is given.
    """
    dimension = gramian.shape[0]
    P = gramian
    q = gramian @ weight_vector
    G = -np.eye(dimension)
    h = np.zeros(dimension)
    return solve_qp(P, q, G, h, solver=solver)


def _to_array(tensor: Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy().astype(np.float64)
