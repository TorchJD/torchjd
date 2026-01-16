from torch import Tensor


def compute_gramian(matrix: Tensor) -> Tensor:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    return matrix @ matrix.T
