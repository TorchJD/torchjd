from typing import TypeGuard

from torch import Tensor


class GeneralizedMatrix(Tensor):
    pass


class Matrix(GeneralizedMatrix):
    pass


class PSDQuadraticForm(Tensor):
    pass


class PSDMatrix(PSDQuadraticForm, Matrix):
    pass


def is_generalized_matrix(t: Tensor) -> TypeGuard[GeneralizedMatrix]:
    return t.ndim >= 1


def is_matrix(t: Tensor) -> TypeGuard[Matrix]:
    return t.ndim == 2


def is_psd_quadratic_form(t: Tensor) -> TypeGuard[PSDQuadraticForm]:
    half_dim = t.ndim // 2
    return t.ndim % 2 == 0 and t.ndim >= 2 and t.shape[:half_dim] == t.shape[: half_dim - 1 : -1]
    # We do not check that t is PSD as it is expensive, but this must be checked in the tests of
    # every function that use this TypeGuard by using `assert_psd_quadratic_form`.


def is_psd_matrix(t: Tensor) -> TypeGuard[PSDMatrix]:
    return t.ndim == 2 and t.shape[0] == t.shape[1]
    # We do not check that t is PSD as it is expensive, but this must be checked in the tests of
    # every function that use this TypeGuard, by using `assert_psd_matrix`.
