import torch
from torch import Tensor
from torch.nn.functional import normalize


def _generate_matrix(m: int, n: int, rank: int) -> Tensor:
    """Generates a random matrix A of shape [m, n] with provided rank."""

    U = _generate_orthonormal_matrix(m)
    Vt = _generate_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U[:, :rank] @ S @ Vt[:rank, :]
    return A


def _generate_strong_stationary_matrix(m: int, n: int) -> Tensor:
    """
    Generates a random matrix A of shape [m, n] with rank min(n, m - 1), such that there exists a
    vector 0<v with v^T A = 0.
    """

    v = torch.abs(torch.randn([m]))
    return _generate_matrix_orthogonal_to_vector(v, n)


def _generate_weak_stationary_matrix(m: int, n: int) -> Tensor:
    """
    Generates a random matrix A of shape [m, n] with rank min(n, m - 1), such that there exists a
    vector 0<=v with one coordinate equal to 0 and such that v^T A = 0.

    Note that if multiple coordinates of v were equal to 0, the generated matrix would still be weak
    stationary, but here we only set one of them to 0 for simplicity.
    """

    v = torch.abs(torch.randn([m]))
    i = torch.randint(0, m, [])
    v[i] = 0.0
    return _generate_matrix_orthogonal_to_vector(v, n)


def _generate_matrix_orthogonal_to_vector(v: Tensor, n: int) -> Tensor:
    """
    Generates a random matrix A of shape [len(v), n] with rank min(n, len(v) - 1) such that
    v^T A = 0.
    """

    rank = min(n, len(v) - 1)
    Q = normalize(v, dim=0).unsqueeze(1)
    U = _generate_semi_orthonormal_complement(Q)
    Vt = _generate_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U[:, :rank] @ S @ Vt[:rank, :]
    return A


def _generate_orthonormal_matrix(dim: int) -> Tensor:
    """Uniformly generates a random orthonormal matrix of shape [dim, dim]."""

    return _generate_semi_orthonormal_complement(torch.zeros([dim, 0]))


def _generate_semi_orthonormal_complement(Q: Tensor) -> Tensor:
    """
    Uniformly generates a random semi-orthonormal matrix Q' (i.e. Q'^T Q' = I) of shape [m, m-k]
    orthogonal to Q, i.e. such that the concatenation [Q, Q'] is an orthonormal matrix.

    :param Q: A semi-orthonormal matrix (i.e. Q^T Q = I) of shape [m, k], with k <= m.
    """

    m, k = Q.shape
    A = torch.randn([m, m - k])

    # project A onto the orthogonal complement of Q
    A_proj = A - Q @ (Q.T @ A)

    Q_prime, _ = torch.linalg.qr(A_proj)
    return Q_prime


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

_scales = [0.0, 1e-10, 1e3, 1e5, 1e10, 1e15]

# Fix seed to fix randomness of matrix generation
torch.manual_seed(0)

matrices = [_generate_matrix(m, n, rank) for m, n, rank in _matrix_dimension_triples]
scaled_matrices = [scale * matrix for scale in _scales for matrix in matrices]
zero_matrices = [torch.zeros([m, n]) for m, n in _zero_matrices_shapes]
matrices_2_plus_rows = [matrix for matrix in matrices + zero_matrices if matrix.shape[0] >= 2]
scaled_matrices_2_plus_rows = [matrix for matrix in scaled_matrices if matrix.shape[0] >= 2]
strong_stationary_matrices = [
    _generate_strong_stationary_matrix(m, n) for m, n in _stationary_matrices_shapes
]
weak_stationary_matrices = strong_stationary_matrices + [
    _generate_weak_stationary_matrix(m, n) for m, n in _stationary_matrices_shapes
]
typical_matrices = zero_matrices + matrices + weak_stationary_matrices + strong_stationary_matrices
