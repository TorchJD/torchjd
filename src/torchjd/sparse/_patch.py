"""Monkey-patch hooks that route sparse ops through TorchJD wrappers.

This module is imported from ``torchjd.sparse`` at import-time.
Patch execution is *idempotent* – calling :pyfunc:`enable_seamless_sparse`
multiple times is safe.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from types import MethodType
from typing import Callable

import torch

from ._autograd import sparse_mm

# The wheel might exist yet be ABI-incompatible with the current
# PyTorch, which raises *OSError* at import-time.

try:  # pragma: no cover
    torch_sparse = import_module("torch_sparse")  # type: ignore
except (ModuleNotFoundError, OSError):
    torch_sparse = None


# Helpers
def _wrap_mm(orig_fn: Callable, wrapper: Callable) -> Callable:
    """Return a patched ``torch.sparse.mm`` that defers to *wrapper*."""

    def patched(A, X):  # noqa: D401
        if isinstance(A, torch.Tensor) and A.is_sparse and X.dim() >= 2:
            return wrapper(A, X)
        return orig_fn(A, X)

    return patched


def _wrap_tensor_matmul(orig_fn: Callable) -> Callable:
    def patched(self, other):  # noqa: D401
        if self.is_sparse and isinstance(other, torch.Tensor) and other.dim() >= 2:
            return sparse_mm(self, other)
        return orig_fn(self, other)

    return patched


# Public API
def enable_seamless_sparse() -> None:
    """Patch common call-sites so users need *no* explicit imports."""
    # torch.sparse.mm
    if getattr(torch.sparse, "_orig_mm", None) is None:
        torch.sparse._orig_mm = torch.sparse.mm  # type: ignore[attr-defined]
        torch.sparse.mm = _wrap_mm(torch.sparse._orig_mm, sparse_mm)  # type: ignore[attr-defined]

    # tensor @ dense
    if getattr(torch.Tensor, "_orig_matmul", None) is None:
        torch.Tensor._orig_matmul = torch.Tensor.__matmul__  # type: ignore[attr-defined]  # noqa: E501
        torch.Tensor.__matmul__ = _wrap_tensor_matmul(
            torch.Tensor._orig_matmul  # type: ignore[attr-defined]
        )  # type: ignore[attr-defined]

    # torch_sparse (optional)
    if torch_sparse is None:
        warnings.warn(
            "torch_sparse not found: SpSpMM will use slow fallback.",
            RuntimeWarning,
            stacklevel=2,
        )  # pragma: no cover
        return

    if not hasattr(torch_sparse.SparseTensor, "_orig_matmul"):

        def _sparse_tensor_matmul(self, dense):  # noqa: D401
            return sparse_mm(self, dense)

        torch_sparse.SparseTensor._orig_matmul = torch_sparse.SparseTensor.matmul  # type: ignore[attr-defined]  # noqa: E501
        torch_sparse.SparseTensor.matmul = MethodType(  # type: ignore[attr-defined]
            _sparse_tensor_matmul, torch_sparse.SparseTensor
        )
