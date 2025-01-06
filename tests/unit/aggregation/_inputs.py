import torch
from torch import Tensor


def _check_valid_dimensions(n_rows: int, n_cols: int) -> None:
    if n_rows < 1:
        raise ValueError(
            f"Parameter `n_rows` should be a positive integer. Found n_rows = {n_rows}."
        )
    if n_cols < 1:
        raise ValueError(
            f"Parameter `n_cols` should be a positive integer. Found n_cols = {n_cols}."
        )


def _check_valid_rank(n_rows: int, n_cols: int, rank: int) -> None:
    if rank < 0:
        raise ValueError(f"Parameter `rank` should be a non-negative integer. Found rank = {rank}.")
    if rank > n_rows:
        raise ValueError(
            "Parameter `rank` should not be larger than the number of rows. "
            f"Found rank = {rank} and n_rows = {n_rows}."
        )
    if rank > n_cols:
        raise ValueError(
            "Parameter `rank` should not be larger than the number of columns. "
            f"Found rank = {rank} and n_cols = {n_cols}."
        )


def _augment_orthogonal_matrix(orthogonal_matrix: Tensor) -> Tensor:
    """
    Augments the provided matrix with one more column that is filled with a random unit vector that
    is orthogonal to the provided orthogonal_matrix.
    """

    n_rows = orthogonal_matrix.shape[0]
    projection = orthogonal_matrix @ orthogonal_matrix.T
    zero = torch.zeros([n_rows])
    while True:
        random_vector = torch.randn([n_rows])
        projected_vector = random_vector - projection @ random_vector
        if not torch.allclose(projected_vector, zero):
            break
    projected_vector = torch.nn.functional.normalize(projected_vector, dim=0).reshape([-1, 1])
    augmented_matrix = torch.cat((orthogonal_matrix, projected_vector), dim=1)
    return augmented_matrix


def _complete_orthogonal_matrix(orthogonal_matrix: Tensor, n_cols: int) -> Tensor:
    """
    Iteratively augments the input ``orthogonal_matrix`` with columns that are orthogonal to its
    existing columns, until it has the required number of columns. Returns the obtained
    orthogonal matrix.
    """

    if orthogonal_matrix.shape[1] > n_cols:
        raise ValueError(
            f"Parameter `n_cols` should exceed the second dimension of the provided matrix. Found "
            f"`n_cols = {n_cols}` and `partial_matrix.shape[1] = {orthogonal_matrix.shape[1]}`."
        )

    for i in range(n_cols - 1):
        orthogonal_matrix = _augment_orthogonal_matrix(orthogonal_matrix)
    return orthogonal_matrix


def _generate_unitary_matrix(n_rows: int, n_cols: int) -> Tensor:
    """Generates a unitary matrix of shape [n_rows, n_cols]."""

    _check_valid_dimensions(n_rows, n_cols)
    partial_matrix = torch.randn([n_rows, 1])
    partial_matrix = torch.nn.functional.normalize(partial_matrix, dim=0)

    unitary_matrix = _complete_orthogonal_matrix(partial_matrix, n_cols)
    return unitary_matrix


def _generate_unitary_matrix_with_positive_column(n_rows: int, n_cols: int) -> Tensor:
    """
    Generates a unitary matrix of shape [n_rows, n_cols] with the first column consisting of an all
    positive vector.
    """
    _check_valid_dimensions(n_rows, n_cols)
    partial_matrix = torch.abs(torch.randn([n_rows, 1]))
    partial_matrix = torch.nn.functional.normalize(partial_matrix, dim=0)

    unitary_matrix_with_positive_column = _complete_orthogonal_matrix(partial_matrix, n_cols)
    return unitary_matrix_with_positive_column


def _generate_diagonal_singular_values(rank: int) -> Tensor:
    """
    generates a diagonal matrix of positive values sorted in descending order.
    """
    singular_values = torch.abs(torch.randn([rank]))
    singular_values = torch.sort(singular_values, descending=True)[0]
    S = torch.diag(singular_values)
    return S


def generate_matrix(n_rows: int, n_cols: int, rank: int) -> Tensor:
    """
    Generates a random matrix of shape [``n_rows``, ``n_cols``] with provided ``rank``.
    """

    _check_valid_rank(n_rows, n_cols, rank)

    if rank == 0:
        matrix = torch.zeros([n_rows, n_cols])
    else:
        U = _generate_unitary_matrix(n_rows, rank)
        V = _generate_unitary_matrix(n_cols, rank)
        S = _generate_diagonal_singular_values(rank)
        matrix = U @ S @ V.T

    return matrix


def generate_stationary_matrix(n_rows: int, n_cols: int, rank: int) -> Tensor:
    """
    Generates a random matrix of shape [``n_rows``, ``n_cols``] with provided ``rank``. The matrix
    has a singular triple (u, s, v) such that u is all (strictly) positive and s is 0.
    """

    _check_valid_rank(n_rows, n_cols, rank)
    if rank == 0:
        matrix = torch.zeros([n_rows, n_cols])
    else:
        U = _generate_unitary_matrix_with_positive_column(n_rows, rank)
        V = _generate_unitary_matrix(n_cols, rank)
        S = _generate_diagonal_singular_values(rank)
        S[0, 0] = 0.0
        matrix = U @ S @ V.T

    return matrix


_matrix_dimension_triples = [
    (1, 1, 1),
    (4, 3, 1),
    (4, 3, 2),
    (4, 3, 3),
    (9, 11, 5),
    (9, 11, 9),
]

_zero_rank_matrix_shapes = [
    (1, 1),
    (4, 3),
    (9, 11),
]

_scales = [0.0, 1e-10, 1.0, 1e3, 1e5, 1e10, 1e15]

# Fix seed to fix randomness of matrix generation
torch.manual_seed(0)

matrices = [
    generate_matrix(n_rows, n_cols, rank) for n_rows, n_cols, rank in _matrix_dimension_triples
]
scaled_matrices = [scale * matrix for scale in _scales for matrix in matrices]
zero_rank_matrices = [torch.zeros([n_rows, n_cols]) for n_rows, n_cols in _zero_rank_matrix_shapes]
matrices_2_plus_rows = [matrix for matrix in matrices + zero_rank_matrices if matrix.shape[0] >= 2]
scaled_matrices_2_plus_rows = [
    matrix for matrix in scaled_matrices + zero_rank_matrices if matrix.shape[0] >= 2
]
stationary_matrices = [
    generate_stationary_matrix(n_rows, n_cols, rank)
    for n_rows, n_cols, rank in _matrix_dimension_triples
]
