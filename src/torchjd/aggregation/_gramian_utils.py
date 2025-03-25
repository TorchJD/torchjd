import torch
from torch import Tensor


def _compute_gramian(matrix: Tensor) -> Tensor:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    return matrix @ matrix.T


def _compute_regularized_normalized_gramian(matrix: Tensor, norm_eps: float, reg_eps: float):
    normalized_gramian = _compute_normalized_gramian(matrix, norm_eps)
    return _regularize(normalized_gramian, reg_eps)


def _compute_normalized_gramian(matrix: Tensor, eps: float) -> Tensor:
    r"""
    Computes :math:`\frac{1}{\sigma_\max^2} J J^T` for an input matrix :math:`J`, where
    :math:`{\sigma_\max^2}` is :math:`J`'s largest singular value.
    .. hint::
        :math:`J J^T` is the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of
        :math:`J`
    For a given matrix :math:`J` with SVD: :math:`J = U S V^T`, we can see that:
    .. math::
        \frac{1}{\sigma_\max^2} J J^T = \frac{1}{\sigma_\max^2} U S V^T V S^T U^T = U
        \left( \frac{S}{\sigma_\max} \right)^2 U^T
    This is the quantity we compute.
    .. note::
        If the provided matrix has dimension :math:`m \times n`, the computation only depends on
        :math:`n` through the SVD algorithm which is efficient, therefore this is rather fast.
    """

    left_unitary_matrix, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    max_singular_value = torch.max(singular_values)
    if max_singular_value < eps:
        scaled_singular_values = torch.zeros_like(singular_values)
    else:
        scaled_singular_values = singular_values / max_singular_value
    normalized_gramian = (
        left_unitary_matrix @ torch.diag(scaled_singular_values**2) @ left_unitary_matrix.T
    )
    return normalized_gramian


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
