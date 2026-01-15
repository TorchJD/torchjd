import torch
from torch import Tensor


def compute_gramian(generalized_matrix: Tensor) -> Tensor:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given
    generalized matrix. Specifically, this is equivalent to

    matrix = generalized_matrix.reshape([generalized_matrix.shape[0], -1])
    return matrix @ matrix.T
    """
    dims = list(range(1, generalized_matrix.ndim))
    gramian = torch.tensordot(generalized_matrix, generalized_matrix, dims=(dims, dims))
    return gramian
