from .matrix import Matrix, PSDMatrix


def compute_gramian(matrix: Matrix) -> PSDMatrix:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    return matrix @ matrix.T
