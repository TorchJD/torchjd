import torch

from torchjd._linalg import PSDMatrix, is_psd_matrix


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
    assert is_psd_matrix(output)
    return output


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
    assert is_psd_matrix(output)
    return output
