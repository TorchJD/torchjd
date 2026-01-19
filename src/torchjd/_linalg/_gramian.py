from ._matrix import Matrix, PSDMatrix, is_psd_matrix


def compute_gramian(matrix: Matrix) -> PSDMatrix:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    gramian = matrix @ matrix.T
    assert is_psd_matrix(gramian)
    return gramian
