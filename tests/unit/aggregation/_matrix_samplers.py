from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn.functional import normalize


class MatrixSampler(ABC):
    """Abstract base class for sampling matrices of a given shape, rank and dtype."""

    def __init__(self, m: int, n: int, rank: int, dtype: torch.dtype = torch.float32):
        self._check_params(m, n, rank, dtype)
        self.m = m
        self.n = n
        self.rank = rank
        self.dtype = dtype

    def _check_params(self, m: int, n: int, rank: int, dtype: torch.dtype) -> None:
        """Checks that the provided __init__ parameters are acceptable."""

        assert m >= 0
        assert n >= 0
        assert 0 <= rank <= min(m, n)
        assert dtype in {torch.float32, torch.float64}

    @abstractmethod
    def __call__(self, rng: torch.Generator | None = None) -> Tensor:
        """Samples a random matrix."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(m={self.m}, n={self.n}, rank={self.rank})"

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__.replace('MatrixSampler', '')}"
            f"({self.m}x{self.n}r{self.rank}:{str(self.dtype)[6:]})"
        )


class NormalSampler(MatrixSampler):
    """Sampler for random normal matrices of shape [m, n] with provided rank and dtype."""

    def __call__(self, rng: torch.Generator | None = None) -> Tensor:
        U = _sample_orthonormal_matrix(self.m, dtype=self.dtype, rng=rng)
        Vt = _sample_orthonormal_matrix(self.n, dtype=self.dtype, rng=rng)
        S = torch.diag(torch.abs(torch.randn([self.rank], dtype=self.dtype, generator=rng)))
        A = U[:, : self.rank] @ S @ Vt[: self.rank, :]
        return A


class StrongSampler(MatrixSampler):
    """
    Sampler for random strongly stationary matrices of shape [m, n] with provided rank and dtype.

    Definition: A matrix A is said to be strongly stationary if there exists a vector 0 < v such
    that v^T A = 0.

    Obtaining such a matrix is done by sampling a positive v, and by then sampling a matrix
    orthogonal to v.
    """

    def _check_params(self, m: int, n: int, rank: int, dtype: torch.dtype) -> None:
        super()._check_params(m, n, rank, dtype)
        assert 1 < m
        assert 0 < rank <= min(m - 1, n)

    def __call__(self, rng: torch.Generator | None = None) -> Tensor:
        v = torch.abs(torch.randn([self.m], dtype=self.dtype, generator=rng))
        U1 = normalize(v, dim=0).unsqueeze(1)
        U2 = _sample_semi_orthonormal_complement(U1, rng=rng)
        Vt = _sample_orthonormal_matrix(self.n, dtype=self.dtype, rng=rng)
        S = torch.diag(torch.abs(torch.randn([self.rank], dtype=self.dtype, generator=rng)))
        A = U2[:, : self.rank] @ S @ Vt[: self.rank, :]
        return A


class StrictlyWeakSampler(MatrixSampler):
    """
    Sampler for random strictly weakly stationary matrices of shape [m, n] with provided rank and
    dtype.

    Definition: A matrix A is said to be weakly stationary if there exists a vector 0 <= v, v != 0,
    such that v^T A = 0.

    Definition: A matrix A is said to be strictly weakly stationary if it is weakly stationary and
    not strongly stationary, i.e. if there exists a vector 0 <= v, v != 0, such that v^T A = 0 and
    there exists no vector 0 < w with w^T A = 0.

    Obtaining such a matrix is done by sampling two unit-norm vectors v, v', whose sum u is a
    positive vector. These two vectors are also non-negative and non-zero, and are furthermore
    orthogonal. Then, a matrix A, orthogonal to v, is sampled. By its orthogonality to v, A is
    weakly stationary. Moreover, since v' is a non-negative left-singular vector of A with positive
    singular value s, any 0 < w satisfies w^T A != 0. Otherwise, we would have
    0 = w^T A A^T v' = s w^T v' > 0, which is a contradiction. A is thus also not strongly
    stationary.
    """

    def _check_params(self, m: int, n: int, rank: int, dtype: torch.dtype) -> None:
        super()._check_params(m, n, rank, dtype)
        assert 1 < m
        assert 0 < rank <= min(m - 1, n)

    def __call__(self, rng: torch.Generator | None = None) -> Tensor:
        u = torch.abs(torch.randn([self.m], dtype=self.dtype, generator=rng))
        split_index = torch.randint(1, self.m, [], generator=rng).item()
        shuffled_range = torch.randperm(self.m, generator=rng)
        v = torch.zeros(self.m, dtype=self.dtype)
        v[shuffled_range[:split_index]] = normalize(u[shuffled_range[:split_index]], dim=0)
        v_prime = torch.zeros(self.m, dtype=self.dtype)
        v_prime[shuffled_range[split_index:]] = normalize(u[shuffled_range[split_index:]], dim=0)
        U1 = torch.stack([v, v_prime]).T
        U2 = _sample_semi_orthonormal_complement(U1, rng=rng)
        U = torch.hstack([U1, U2])
        Vt = _sample_orthonormal_matrix(self.n, dtype=self.dtype, rng=rng)
        S = torch.diag(torch.abs(torch.randn([self.rank], dtype=self.dtype, generator=rng)))
        A = U[:, 1 : self.rank + 1] @ S @ Vt[: self.rank, :]
        return A


class NonWeakSampler(MatrixSampler):
    """
    Sampler for a random non weakly-stationary matrices of shape [m, n] with provided rank and
    dtype.

    Obtaining such a matrix is done by sampling a positive u, and by then sampling a matrix A that
    has u as one of its left-singular vectors, with positive singular value s. Any 0 <= v, v != 0,
    satisfies v^T A != 0. Otherwise, we would have 0 = v^T A A^T u = s v^T u > 0, which is a
    contradiction. A is thus not weakly stationary.
    """

    def _check_params(self, m: int, n: int, rank: int, dtype: torch.dtype) -> None:
        super()._check_params(m, n, rank, dtype)
        assert 0 < rank

    def __call__(self, rng: torch.Generator | None = None) -> Tensor:
        u = torch.abs(torch.randn([self.m], dtype=self.dtype, generator=rng))
        U1 = normalize(u, dim=0).unsqueeze(1)
        U2 = _sample_semi_orthonormal_complement(U1, rng=rng)
        U = torch.hstack([U1, U2])
        Vt = _sample_orthonormal_matrix(self.n, dtype=self.dtype, rng=rng)
        S = torch.diag(torch.abs(torch.randn([self.rank], dtype=self.dtype, generator=rng)))
        A = U[:, : self.rank] @ S @ Vt[: self.rank, :]
        return A


def _sample_orthonormal_matrix(
    dim: int, dtype: torch.dtype, rng: torch.Generator | None = None
) -> Tensor:
    """Uniformly samples a random orthonormal matrix of shape [dim, dim]."""

    return _sample_semi_orthonormal_complement(torch.zeros([dim, 0], dtype=dtype), rng=rng)


def _sample_semi_orthonormal_complement(Q: Tensor, rng: torch.Generator | None = None) -> Tensor:
    """
    Uniformly samples a random semi-orthonormal matrix Q' (i.e. Q'^T Q' = I) of shape [m, m-k]
    orthogonal to Q, i.e. such that the concatenation [Q, Q'] is an orthonormal matrix.

    :param Q: A semi-orthonormal matrix (i.e. Q^T Q = I) of shape [m, k], with k <= m.
    """

    dtype = Q.dtype
    m, k = Q.shape
    A = torch.randn([m, m - k], dtype=dtype, generator=rng)

    # project A onto the orthogonal complement of Q
    A_proj = A - Q @ (Q.T @ A)

    Q_prime, _ = torch.linalg.qr(A_proj)
    return Q_prime
