import torch
from torch import Tensor
from torch.linalg import LinAlgError


def _compute_gramian(matrix: Tensor) -> Tensor:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """
    return matrix @ matrix.T


def _compute_normalized_gramian(matrix: Tensor, norm_eps: float) -> Tensor:
    r"""
    Computes :math:`\frac{1}{\sigma_\max^2} A A^T` for an input matrix :math:`A`, where
    :math:`{\sigma_\max^2}` is :math:`A`'s largest singular value.
    .. hint::
        :math:`A A^T` is the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of
        :math:`A`
    For a given matrix :math:`A` with SVD: :math:`A = U S V^T`, we can see that:
    .. math::
        \frac{1}{\sigma_\max^2} A A^T = \frac{1}{\sigma_\max^2} U S V^T V S^T U^T = U
        \left( \frac{S}{\sigma_\max} \right)^2 U^T
    This is the quantity we compute.
    .. note::
        If the provided matrix has dimension :math:`m \times n`, the computation only depends on
        :math:`n` through the SVD algorithm which is efficient, therefore this is rather fast.
    """

    try:
        left_unitary_matrix, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    except LinAlgError as error:  # Not sure if this can happen
        raise ValueError(
            f"Unexpected failure of the svd computation on matrix {matrix}. Please open an "
            "issue on https://github.com/TorchJD/torchjd/issues and paste this error message in it."
        ) from error
    max_singular_value = torch.max(singular_values)
    if max_singular_value < norm_eps:
        scaled_singular_values = torch.zeros_like(singular_values)
    else:
        scaled_singular_values = singular_values / max_singular_value
    normalized_gramian = (
        left_unitary_matrix @ torch.diag(scaled_singular_values**2) @ left_unitary_matrix.T
    )
    return normalized_gramian
