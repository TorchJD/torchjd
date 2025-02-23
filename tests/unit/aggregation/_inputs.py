import torch
from torch import Tensor


def generate_orthogonal_matrix(dim: int) -> Tensor:
    """
    Uniformly generates a random orthogonal matrix of shape [n, n].
    """

    A = torch.randn([dim, dim])
    Q, _ = torch.qr(A)
    return Q


def complete_orthogonal_matrix(vector: Tensor) -> Tensor:
    """
    Uniformly generates a random orthogonal matrix of shape [len(vector), len(vector)] such that the
    first column is the normalization of the provided vector.
    """

    n = vector.shape[0]
    u = torch.nn.functional.normalize(vector, dim=0)
    A = torch.randn([n, n - 1])

    # project A onto the orthogonal complement of u
    A_proj = A - u.unsqueeze(1) * (u.unsqueeze(0) @ A)

    Q, _ = torch.qr(A_proj)
    return torch.cat([u.unsqueeze(1), Q], dim=1)


def generate_matrix(n_rows: int, n_cols: int, rank: int) -> Tensor:
    """
    Generates a random matrix of shape [``n_rows``, ``n_cols``] with provided ``rank``.
    """

    U = generate_orthogonal_matrix(n_rows)
    Vt = generate_orthogonal_matrix(n_cols)
    S = torch.diag(torch.abs(torch.randn([rank])))
    matrix = U[:, :rank] @ S @ Vt[:rank, :]
    return matrix


def generate_matrix_with_orthogonal_vector(vector: Tensor, n_cols: int, rank: int) -> Tensor:
    """
    Generates a random matrix of shape [``len(vector)``, ``n_cols``] with rank
    ``min(rank, len(vector)-1)``. Such that `vector @ matrix` is zero.
    """

    n_rows = len(vector)
    effective_rank = min(rank, n_rows - 1)
    U = complete_orthogonal_matrix(vector)
    Vt = generate_orthogonal_matrix(n_cols)
    S = torch.diag(torch.abs(torch.randn([effective_rank])))
    matrix = U[:, 1 : 1 + effective_rank] @ S @ Vt[:effective_rank, :]
    return matrix


def generate_strong_stationary_matrix(n_rows: int, n_cols: int, rank: int) -> Tensor:
    """
    Generates a random matrix of shape [``n_rows``, ``n_cols``] with rank
    ``min(rank, len(vector)-1)``, such that there exists a vector `0<v` with `v @ matrix=0`.
    """
    v = torch.abs(torch.randn([n_rows]))
    return generate_matrix_with_orthogonal_vector(v, n_cols, rank)


def generate_weak_stationary_matrix(n_rows: int, n_cols: int, rank: int) -> Tensor:
    """
    Generates a random matrix of shape [``n_rows``, ``n_cols``] with rank
    ``min(rank, len(vector)-1)``, such that there exists a vector `0<=v` with at least one
    coordinate equal to `0` and such that `v @ matrix=0`.
    """
    v = torch.abs(torch.randn([n_rows]))
    v[torch.randint(0, n_rows, [])] = 0.0
    return generate_matrix_with_orthogonal_vector(v, n_cols, rank)


_matrix_dimension_triples = [
    (1, 1, 1),
    (4, 3, 1),
    (4, 3, 2),
    (4, 3, 3),
    (9, 11, 5),
    (9, 11, 9),
]

_zero_matrices_shapes = [
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
zero_matrices = [torch.zeros([n_rows, n_cols]) for n_rows, n_cols in _zero_matrices_shapes]
matrices_2_plus_rows = [matrix for matrix in matrices + zero_matrices if matrix.shape[0] >= 2]
scaled_matrices_2_plus_rows = [
    matrix for matrix in scaled_matrices + zero_matrices if matrix.shape[0] >= 2
]
strong_stationary_matrices = [
    generate_strong_stationary_matrix(n_rows, n_cols, rank)
    for n_rows, n_cols, rank in _matrix_dimension_triples
]
