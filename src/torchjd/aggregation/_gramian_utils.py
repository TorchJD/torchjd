import torch
from torch import Tensor


def _compute_gramian(matrix: Tensor) -> Tensor:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    return matrix @ matrix.T


def _compute_regularized_gramian(matrix: Tensor, reg_eps: float):
    gramian = _compute_gramian(matrix)
    return _regularize(gramian, reg_eps)


def _regularize(gramian: Tensor, eps: float) -> Tensor:
    """
    Adds a regularization term to the gramian to enforce positive definiteness.

    Because of numerical errors, `gramian` might have slightly negative eigenvalue(s). Adding a
    regularization term which is a small proportion of the identity matrix ensures that the gramian
    is positive definite.
    """

    max_singular_value = torch.max(torch.linalg.svdvals(gramian))

    if max_singular_value < 0.0001:
        return torch.zeros_like(gramian) + eps * torch.eye(
            gramian.shape[0], dtype=gramian.dtype, device=gramian.device
        )

    regularization_matrix = (
        eps
        * (max_singular_value)
        * torch.eye(gramian.shape[0], dtype=gramian.dtype, device=gramian.device)
    )
    return gramian + regularization_matrix
