from typing import TypeGuard

from torch import Tensor


class Matrix(Tensor):
    """Tensor with exactly 2 dimensions."""


class PSDGeneralizedMatrix(Tensor):
    """
    Tensor representing a quadratic form. The first half of its dimensions matches the reversed
    second half of its dimensions (e.g. shape=[4, 3, 3, 4]), and its reshaping into a matrix should
    be positive semi-definite.
    """


class PSDMatrix(PSDGeneralizedMatrix, Matrix):
    """Positive semi-definite matrix."""


def is_matrix(t: Tensor) -> TypeGuard[Matrix]:
    return t.ndim == 2


def is_psd_generalized_matrix(t: Tensor) -> TypeGuard[PSDGeneralizedMatrix]:
    half_dim = t.ndim // 2
    return t.ndim % 2 == 0 and t.shape[:half_dim] == t.shape[: half_dim - 1 : -1]
    # We do not check that t is PSD as it is expensive, but this must be checked in the tests of
    # every function that uses this TypeGuard by using `assert_psd_generalized_matrix`.


def is_psd_matrix(t: Tensor) -> TypeGuard[PSDMatrix]:
    return t.ndim == 2 and t.shape[0] == t.shape[1]
    # We do not check that t is PSD as it is expensive, but this must be checked in the tests of
    # every function that uses this TypeGuard, by using `assert_psd_matrix`.
