import torch
from torch import Tensor


def _compute_gramian(matrix: Tensor) -> Tensor:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    return matrix @ matrix.T


def _compute_normalized_gramian(matrix: Tensor, eps: float) -> Tensor:
    gramian = _compute_gramian(matrix)
    return _normalize(gramian, eps)


def _normalize(gramian: Tensor, eps: float) -> Tensor:
    """
    Normalizes the gramian with respect to the Frobenius norm.

    If `G=A A^T`, then the Frobenius norm of `A` is the square root of the trace of `G`, i.e., the
    sqrt of the sum of the diagonal elements. The gramian of the (Frobenius) normalization of `A` is
    therefore `G` divided by the sum of its diagonal elements.
    """
    squared_frobenius_norm = gramian.diagonal().sum()
    if squared_frobenius_norm < eps:
        return torch.zeros_like(gramian)
    else:
        return gramian / squared_frobenius_norm


def _regularize(gramian: Tensor, eps: float) -> Tensor:
    """
    Adds a regularization term to the gramian to enforce positive definiteness.

    Because of numerical errors, `gramian` might have slightly negative eigenvalue(s). Adding a
    regularization term which is a small proportion of the identity matrix ensures that the gramian
    is positive definite.
    """

    regularization_matrix = eps * torch.eye(
        gramian.shape[0], dtype=gramian.dtype, device=gramian.device
    )
    return gramian + regularization_matrix
