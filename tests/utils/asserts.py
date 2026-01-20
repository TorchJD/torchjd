import torch
from torch.testing import assert_close

from torchjd._linalg import PSDGeneralizedMatrix, PSDMatrix
from torchjd.autogram._gramian_utils import flatten
from torchjd.autojac._accumulation import is_tensor_with_jac


def assert_has_jac(t: torch.Tensor) -> None:
    assert is_tensor_with_jac(t)
    assert t.jac is not None and t.jac.shape[1:] == t.shape


def assert_has_no_jac(t: torch.Tensor) -> None:
    assert not is_tensor_with_jac(t)


def assert_jac_close(t: torch.Tensor, expected_jac: torch.Tensor, **kwargs) -> None:
    assert is_tensor_with_jac(t)
    assert_close(t.jac, expected_jac, **kwargs)


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


def assert_psd_generalized_matrix(t: PSDGeneralizedMatrix, **kwargs) -> None:
    matrix = flatten(t)
    assert_psd_matrix(matrix, **kwargs)
