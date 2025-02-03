from typing import Literal

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor


def _get_projection_weights(
    gramian: Tensor, weights: Tensor, solver: Literal["quadprog"]
) -> Tensor:
    r"""
    Computes the projection weights of a tensor of weights onto the dual cone of a matrix, given
    its Gramian matrix.

    Specifically, as stated in Proposition 1 of [1], let:
    - `J` be a matrix,
    - `G = J J^\top` its Gramian,
    - `u` a vector.

    The projection of `J^\top u` onto the dual cone of `J` is `J^\top w`, where `w` is the solution
    to the optimization problem:

        minimize        v^\top G v
        subject to      u \preceq v

    This function calculates `w` when provided with `G` and `u`.

    Reference:
    [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    **Note:** This function is tensorized, meaning `weights` can be a tensor with additional batch
    dimensions rather than a single vector.


    :param gramian: The Gramian matrix with shape `[n, n]`.
    :param weights: A tensor of weights to be projected with shape `[*, n]`.
    :param solver: The quadratic programming solver to use.
    :return: A tensor of projection weights with shape `[*, n]`, corresponding to `weights`.
    """

    weight_matrix = _to_array(weights.reshape([-1, weights.shape[-1]]))
    gramian_array = _to_array(gramian)

    lagrange_multiplier_vectors = [
        _get_projection_weight_vector(gramian_array, weight_vector, solver)
        for weight_vector in weight_matrix
    ]

    lagrange_multiplier_matrix = np.stack(lagrange_multiplier_vectors)
    lagrange_multipliers = (
        torch.from_numpy(lagrange_multiplier_matrix)
        .to(device=gramian.device, dtype=gramian.dtype)
        .reshape(weights.shape)
    )
    return lagrange_multipliers


def _get_projection_weight_vector(
    gramian: np.array, weight_vector: np.array, solver: Literal["quadprog"]
) -> np.array:
    r"""
    Solves for `v` the quadratic problem

        minimize        v^\top G v
        subject to      u \preceq v

    with `G=gramian` and `u=weight_vector`.
    """

    dimension = gramian.shape[0]
    P = gramian
    q = np.zeros(dimension)
    G = -np.eye(dimension)
    h = -weight_vector
    return solve_qp(P, q, G, h, solver=solver)


def _to_array(tensor: Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy().astype(np.float64)
