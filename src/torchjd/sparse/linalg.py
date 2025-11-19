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
