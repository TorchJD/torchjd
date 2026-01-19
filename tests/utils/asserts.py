from typing import cast

import torch
from torch.testing import assert_close

from torchjd._linalg import PSDMatrix
from torchjd.autojac._accumulation import TensorWithJac


def assert_has_jac(t: torch.Tensor) -> None:
    assert hasattr(t, "jac")
    t_ = cast(TensorWithJac, t)
    assert t_.jac is not None and t_.jac.shape[1:] == t_.shape


def assert_has_no_jac(t: torch.Tensor) -> None:
    assert not hasattr(t, "jac")


def assert_jac_close(t: torch.Tensor, expected_jac: torch.Tensor, **kwargs) -> None:
    assert hasattr(t, "jac")
    t_ = cast(TensorWithJac, t)
    assert_close(t_.jac, expected_jac, **kwargs)


def assert_has_grad(t: torch.Tensor) -> None:
    assert (t.grad is not None) and (t.shape == t.grad.shape)


def assert_has_no_grad(t: torch.Tensor) -> None:
    assert t.grad is None


def assert_grad_close(t: torch.Tensor, expected_grad: torch.Tensor, **kwargs) -> None:
    assert t.grad is not None
    assert_close(t.grad, expected_grad, **kwargs)


def assert_psd_matrix(matrix: PSDMatrix, **kwargs) -> None:

    assert_close(matrix, matrix.mH, **kwargs, msg="Matrix is not symmetric/Hermitian")

    eig_vals = torch.linalg.eigvalsh(matrix)
    expected_eig_vals = eig_vals.clamp(min=0.0)

    assert_close(
        eig_vals, expected_eig_vals, **kwargs, msg="Matrix has significant negative eigenvalues"
    )
