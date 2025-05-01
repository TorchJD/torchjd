import torch
from torch import Tensor


def project_weights(U: Tensor, G: Tensor, max_iter: int, eps: float) -> Tensor:
    """
    Computes the tensor of weights corresponding to the projection of the vectors in `U` onto the
    rows of a matrix whose Gramian is provided.

    This is a tensorization of _project_weight_matrix

    :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    :param max_iter: The maximum number of steps of projected gradient descent.
    :param eps: Tolerance precision threshold to stop optimizing. A lower value leads to a higher
        precision but a potentially larger number of iterations.
    :return: A tensor of projection weights with the same shape as `U`.
    """

    shape = U.shape
    m = shape[-1]
    U_matrix = U.reshape([-1, m]).T

    V = _project_weight_matrix(U_matrix, G, max_iter, eps)

    return V.T.reshape(shape)


def _project_weight_matrix(U: Tensor, G: Tensor, max_iter: int, eps: float) -> Tensor:
    r"""
    Computes the tensor of weights corresponding to the projection of the columns in `U` onto the
    rows of a matrix whose Gramian is provided.


    By Proposition 1 of [1], this is equivalent to solving for `v` the following quadratic program:

    minimize        f(v)=\frac{1}{2} v^T G v
    subject to      u \preceq v

    for each column u in U.

    This is done by projected gradient descent:
    - Initialize at $v_0=u$.
    - At step t, let $w_t = v_{t-1} - \gamma \nabla f(v_{t-1})=v_{t-1}-\gamma G v_{t-1}$
    - let $v_t$ be the projection of $w_t$ onto the feasible cone $\{ v: u \preceq v\}$, i.e.,
      $w_t = \max(u, w_t)$ coordinate-wise.
    - If $v_{t+1}-v_t$ is small, or if we reached the maximal number of iteration, return $v_t$.

    Let $\lambda$ be the maximal eigen-value of $G$. The typical step-size  $\gamma$ should be in
    $]0, 2/\lambda[$ with some theoretical guarantees at $1/\lambda$. We pick a rather aggressive
    step-size $\gamma=1.9/\lambda$ as it works well in practice.

    Reference:
    [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param U: The matrix of weight vectors corresponding to the vectors to project, of shape
        `[m, k]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    :param max_iter: The maximal number of iterations of the solver.
    :param eps: Tolerance precision threshold to stop optimizing. A lower value leads to a higher
        precision but a potentially larger number of iterations.
    :return: A tensor of projection weights with the same shape as `U`.
    """
    driver = "gesvdj" if G.device.type == "cuda" else None
    lambda_max = torch.linalg.svd(G, driver=driver)[1][0]
    if lambda_max < 1e-10:
        return U

    step_size = 1.9 / lambda_max

    V = U.clone()
    for t in range(1, max_iter + 1):
        V_new = torch.maximum(V - step_size * (G @ V), U)
        if (V - V_new).norm() < eps:
            break
        V = V_new
    return V
