import torch
from torch import Tensor


def _generate_matrix(m: int, n: int, rank: int) -> Tensor:
    """Generates a random matrix A of shape [``m``, ``n``] with provided ``rank``."""

    U = _generate_orthonormal_matrix(m)
    Vt = _generate_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U[:, :rank] @ S @ Vt[:rank, :]
    return A


def _generate_strong_stationary_matrix(m: int, n: int) -> Tensor:
    """
    Generates a random matrix A of shape [``m``, ``n``] with rank ``min(n, m - 1)``, such that there
    exists a vector ``0<v`` with ``v @ A = 0``.
    """

    v = torch.abs(torch.randn([m]))
    return _generate_matrix_orthogonal_to_vector(v, n)


def _generate_weak_stationary_matrix(m: int, n: int) -> Tensor:
    """
    Generates a random matrix A of shape [``m``, ``n``] with rank ``min(n, m - 1)``, such that there
    exists a vector ``0<=v`` with at least one coordinate equal to ``0`` and such that
    ``v @ A = 0``.
    """

    v = torch.abs(torch.randn([m]))
    v[torch.randint(0, m, [])] = 0.0
    return _generate_matrix_orthogonal_to_vector(v, n)


def _generate_orthonormal_matrix(dim: int) -> Tensor:
    """Uniformly generates a random orthonormal matrix Q of shape [``dim``, ``dim``]."""

    A = torch.randn([dim, dim])
    Q, _ = torch.linalg.qr(A)
    return Q


def _generate_matrix_orthogonal_to_vector(v: Tensor, n: int) -> Tensor:
    """
    Generates a random matrix A of shape [``len(v)``, ``n``] with rank ``min(n, len(v) - 1)`` such
    that ``v @ A = 0``.
    """

    m = len(v)
    rank = min(n, m - 1)
    U = _generate_semi_orthonormal_matrix_orthogonal_to_vector(v, n=rank)
    Vt = _generate_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U @ S @ Vt[:rank, :]
    return A


def _generate_semi_orthonormal_matrix_orthogonal_to_vector(v: Tensor, n: int) -> Tensor:
    """
    Uniformly generates a random semi-orthonormal matrix Q of shape [``len(v)``, ``n``] such that
    ``v @ Q = 0``.
    """

    m = v.shape[0]
    u = torch.nn.functional.normalize(v, dim=0)
    A = torch.randn([m, n])

    # project A onto the orthogonal complement of u
    A_proj = A - u.unsqueeze(1) * (u.unsqueeze(0) @ A)

    Q, _ = torch.linalg.qr(A_proj)
    return Q


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

_stationary_matrices_shapes = [
    (5, 3),
    (9, 11),
]

_scales = [0.0, 1e-10, 1.0, 1e3, 1e5, 1e10, 1e15]

# Fix seed to fix randomness of matrix generation
torch.manual_seed(0)

matrices = [_generate_matrix(m, n, rank) for m, n, rank in _matrix_dimension_triples]
scaled_matrices = [scale * matrix for scale in _scales for matrix in matrices]
zero_matrices = [torch.zeros([m, n]) for m, n in _zero_matrices_shapes]
matrices_2_plus_rows = [matrix for matrix in matrices + zero_matrices if matrix.shape[0] >= 2]
scaled_matrices_2_plus_rows = [
    matrix for matrix in scaled_matrices + zero_matrices if matrix.shape[0] >= 2
]
strong_stationary_matrices = [
    _generate_strong_stationary_matrix(m, n) for m, n in _stationary_matrices_shapes
]
weak_stationary_matrices = strong_stationary_matrices + [
    _generate_weak_stationary_matrix(m, n) for m, n in _stationary_matrices_shapes
]
