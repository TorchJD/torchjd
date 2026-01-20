from typing import cast

import torch

from ._matrix import GeneralizedMatrix, PSDMatrix


def compute_gramian(matrix: GeneralizedMatrix) -> PSDMatrix:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    indices = list(range(1, matrix.ndim))
    gramian = torch.tensordot(matrix, matrix, dims=(indices, indices))
    return cast(PSDMatrix, gramian)


def normalize(gramian: PSDMatrix, eps: float) -> PSDMatrix:
    """
    Normalizes the gramian `G=AA^T` with respect to the Frobenius norm of `A`.

    If `G=A A^T`, then the Frobenius norm of `A` is the square root of the trace of `G`, i.e., the
    sqrt of the sum of the diagonal elements. The gramian of the (Frobenius) normalization of `A` is
    therefore `G` divided by the sum of its diagonal elements.
    """
    squared_frobenius_norm = gramian.diagonal().sum()
    if squared_frobenius_norm < eps:
        output = torch.zeros_like(gramian)
    else:
        output = gramian / squared_frobenius_norm
    return cast(PSDMatrix, output)


def regularize(gramian: PSDMatrix, eps: float) -> PSDMatrix:
    """
    Adds a regularization term to the gramian to enforce positive definiteness.

    Because of numerical errors, `gramian` might have slightly negative eigenvalue(s). Adding a
    regularization term which is a small proportion of the identity matrix ensures that the gramian
    is positive definite.
    """

    regularization_matrix = eps * torch.eye(
        gramian.shape[0], dtype=gramian.dtype, device=gramian.device
    )
    output = gramian + regularization_matrix
    return cast(PSDMatrix, output)
