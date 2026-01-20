from typing import cast

from ._matrix import Matrix, PSDMatrix


def compute_gramian(matrix: Matrix) -> PSDMatrix:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    gramian = matrix @ matrix.T
    return cast(PSDMatrix, gramian)
