import torch
from torch import Tensor


def project_weights(U: Tensor, G: Tensor, max_iter: int = 200, eps: float = 1e-07) -> Tensor:
    r"""
    Computes the tensor of weights corresponding to the projection of the vectors in `U` onto the
    rows of a matrix whose Gramian is provided.


    By Proposition 1 of [1], this is equivalent to solving for `v` the following quadratic program:

    minimize        v^T G v
    subject to      u \preceq v

    for each u in U.

    This is done by projected gradient descent.

    Reference:
    [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    :param max_iter: The maximum number of step of projected gradient descent.
    :param eps: Tolerance precision threshold to stop optimizing.
    :return: A tensor of projection weights with the same shape as `U`.
    """

    shape = U.shape
    m = shape[-1]
    U_matrix = U.reshape([-1, m]).T
    V = U_matrix.clone()

    # torch.linalg.eigvals synchronizes G on the CPU.
    lambda_max = torch.max(torch.linalg.eigvals(G).real).item()
    for t in range(1, max_iter + 1):
        sigma = 1.0 / t**0.5
        step_size = 2.0 / (lambda_max + sigma)
        V_new = torch.maximum(V - step_size * (G @ V), U_matrix)
        gap = (V - V_new).norm()
        if gap < eps:
            break
        V = V_new
    return V.T.reshape(shape)
