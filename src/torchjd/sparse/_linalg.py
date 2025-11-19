import torch
from torch import Tensor


def solve_int(A: Tensor, B: Tensor, tol=1e-9) -> Tensor | None:
    """
    Solve A X = B where A, B and X have integer dtype.
    Return X if such a matrix exists and otherwise None.
    """

    A_ = A.to(torch.float64)
    B_ = B.to(torch.float64)

    try:
        X = torch.linalg.solve(A_, B_)
    except RuntimeError:
        return None

    X_rounded = X.round()
    if not torch.all(torch.isclose(X, X_rounded, atol=tol)):
        return None

    # TODO: Verify that the round operation cannot fail
    return X_rounded.to(torch.int64)


def mod_c(t1: Tensor, t2: Tensor) -> Tensor:
    """
    Computes the combined modulo r = t1 %c t2, such that
    t1 = d * t2 + r with d = t1 //c t2 and
    0 <= r[i] <= t1[i] for all i.

    :param t1: Non-negative integer vector.
    :param t2: Non-negative integer vector.

    Examples:
        [8, 12]^T %c [2, 3]^T = [0, 0]^T
        [8, 12]^T %c [2, 4]^T = [2, 0]^T
        [8, 12]^T %c [3, 3]^T = [2, 6]^T
        [8, 12]^T %c [2, 0]^T = [0, 12]^T
        [8, 12]^T %c [0, 2]^T = [8, 0]^T
        [8, 12]^T %c [0, 0]^T => ZeroDivisionError
    """

    return t1 - intdiv_c(t1, t2) * t2


def intdiv_c(t1: Tensor, t2: Tensor) -> Tensor:
    """
    Computes the combined integer division d = t1 // t2, such that
    t1 = d * t2 + r with r = t1 %c t2
    0 <= r[i] <= t1[i] for all i.

    :param t1: Non-negative integer vector.
    :param t2: Non-negative integer vector.

    Examples:
        [8, 12]^T //c [2, 3]^T = 4
        [8, 12]^T //c [2, 4]^T = 3
        [8, 12]^T //c [3, 3]^T = 2
        [8, 12]^T //c [2, 0]^T = 4
        [8, 12]^T //c [0, 2]^T = 6
        [8, 12]^T //c [0, 0]^T => ZeroDivisionError
    """

    non_zero_indices = torch.nonzero(t2)
    if len(non_zero_indices) == 0:
        raise ZeroDivisionError("division by zero")
    else:
        min_divider = (t1[non_zero_indices] // t2[non_zero_indices]).min()
        return min_divider
