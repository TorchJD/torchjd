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


def _generate_strong_matrix(m: int, n: int, rank: int) -> Tensor:
    """
    Generates a random strongly stationary matrix A of shape [m, n] with provided rank.

    Definition: A matrix A is said to be strongly stationary if there exists a vector 0 < v such
    that v^T A = 0.

    This is done by generating a positive v, and by then generating a matrix orthogonal to v.
    """

    assert 1 < m
    assert 0 < rank <= min(m - 1, n)

    v = torch.abs(torch.randn([m]))
    U1 = normalize(v, dim=0).unsqueeze(1)
    U2 = _generate_semi_orthonormal_complement(U1)
    Vt = _generate_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U2[:, :rank] @ S @ Vt[:rank, :]
    return A


def _generate_strictly_weak_matrix(m: int, n: int, rank: int) -> Tensor:
    """
    Generates a random strictly weakly stationary matrix A of shape [m, n] with provided rank.

    Definition: A matrix A is said to be weakly stationary if there exists a vector 0 <= v, v != 0,
    such that v^T A = 0.

    Definition: A matrix A is said to be strictly weakly stationary if it is weakly stationary and
    not strongly stationary, i.e. if there exists a vector 0 <= v, v != 0, such that v^T A = 0 and
    there exists no vector 0 < w with w^T A = 0.

    This is done by generating two unit-norm vectors v, v', whose sum u is a positive vector. These
    two vectors are also non-negative and non-zero, and are furthermore orthogonal. Then, a matrix
    A, orthogonal to v, is generated. By its orthogonality to v, A is weakly stationary. Moreover,
    since v' is a non-negative left-singular vector of A with positive singular value s, any 0 < w
    satisfies w^T A != 0. Otherwise, we would have 0 = w^T A A^T v' = s w^T v' > 0, which is a
    contradiction. A is thus also not strongly stationary.
    """

    assert 1 < m
    assert 0 < rank <= min(m - 1, n)

    u = torch.abs(torch.randn([m]))
    split_index = torch.randint(1, m, []).item()
    shuffled_range = torch.randperm(m)
    v = torch.zeros(m)
    v[shuffled_range[:split_index]] = normalize(u[shuffled_range[:split_index]], dim=0)
    v_prime = torch.zeros(m)
    v_prime[shuffled_range[split_index:]] = normalize(u[shuffled_range[split_index:]], dim=0)
    U1 = torch.stack([v, v_prime]).T
    U2 = _generate_semi_orthonormal_complement(U1)
    U = torch.hstack([U1, U2])
    Vt = _generate_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U[:, 1 : rank + 1] @ S @ Vt[:rank, :]
    return A


def _generate_non_weak_matrix(m: int, n: int, rank: int) -> Tensor:
    """
    Generates a random non weakly-stationary matrix A of shape [m, n] with provided rank.

    This is done by generating a positive u, and by then generating a matrix A that has u as one of
    its left-singular vectors, with positive singular value s. Any 0 <= v, v != 0, satisfies
    v^T A != 0. Otherwise, we would have 0 = v^T A A^T u = s v^T u > 0, which is a contradiction. A
    is thus not weakly stationary.
    """

    assert 0 < rank <= min(m, n)

    u = torch.abs(torch.randn([m]))
    U1 = normalize(u, dim=0).unsqueeze(1)
    U2 = _generate_semi_orthonormal_complement(U1)
    U = torch.hstack([U1, U2])
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


_normal_dims = [
    (1, 1, 1),
    (4, 3, 1),
    (4, 3, 2),
    (4, 3, 3),
    (9, 11, 5),
    (9, 11, 9),
]

_zero_dims = [
    (1, 1, 0),
    (4, 3, 0),
    (9, 11, 0),
]

_stationarity_dims = [
    (50, 10, 10),
    (50, 10, 5),
    (50, 10, 1),
    (50, 500, 1),
    (50, 500, 49),
]

_scales = [0.0, 1e-10, 1e3, 1e5, 1e10, 1e15]

# Fix seed to fix randomness of matrix generation
torch.manual_seed(0)

matrices = [_generate_matrix(m, n, r) for m, n, r in _normal_dims]
zero_matrices = [torch.zeros([m, n]) for m, n, _ in _zero_dims]
strong_matrices = [_generate_strong_matrix(m, n, r) for m, n, r in _stationarity_dims]
strictly_weak_matrices = [_generate_strictly_weak_matrix(m, n, r) for m, n, r in _stationarity_dims]
non_weak_matrices = [_generate_non_weak_matrix(m, n, r) for m, n, r in _stationarity_dims]

scaled_matrices = [scale * matrix for scale in _scales for matrix in matrices]

non_strong_matrices = strictly_weak_matrices + non_weak_matrices
typical_matrices = zero_matrices + matrices + strong_matrices + non_strong_matrices

scaled_matrices_2_plus_rows = [matrix for matrix in scaled_matrices if matrix.shape[0] >= 2]
typical_matrices_2_plus_rows = [matrix for matrix in typical_matrices if matrix.shape[0] >= 2]
