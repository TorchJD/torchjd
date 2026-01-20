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
