"""Vmap-compatible sparse @ dense for TorchJD."""

from __future__ import annotations

from typing import Tuple

import torch

from ._registry import to_coalesced_coo

_orig_sparse_mm = getattr(torch.sparse, "_orig_mm", torch.sparse.mm)


class _SparseMatMul(torch.autograd.Function):
    """y = A @ X where **A** is sparse and **X** is dense."""

    @staticmethod
    def forward(A_like: torch.Tensor, X: torch.Tensor) -> torch.Tensor:  # noqa: D401
        A = to_coalesced_coo(A_like)

        if X.dim() == 3:  # (B, N, d)
            B, N, d = X.shape
            X2d = X.reshape(B * N, d).view(N, B * d)
            Y2d = _orig_sparse_mm(A, X2d)  # pragma: no cover
            return Y2d.view(N, B, d).permute(1, 0, 2)  # pragma: no cover

        return _orig_sparse_mm(A, X)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:  # noqa: D401
        A_like, _ = inputs
        ctx.save_for_backward(to_coalesced_coo(A_like))

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> Tuple[None, torch.Tensor]:
        (A,) = ctx.saved_tensors
        AT = A.transpose(0, 1)

        if dY.dim() == 3:  # batched
            B, N, d = dY.shape
            dY2d = dY.permute(1, 0, 2).reshape(N, B * d)
            dX2d = _orig_sparse_mm(AT, dY2d)
            dX = dX2d.view(N, B, d).permute(1, 0, 2)
            return None, dX

        return None, _orig_sparse_mm(AT, dY)  # pragma: no cover

    @staticmethod
    def vmap(info, in_dims, A_unbatched, X_batched):  # noqa: D401
        A = A_unbatched  # shared
        X = X_batched  # (B, N, d)

        B, N, d = X.shape
        X2d = X.reshape(B * N, d).view(N, B * d)
        Y2d = _orig_sparse_mm(A, X2d)
        Y = Y2d.view(N, B, d).permute(1, 0, 2)
        return Y, 0  # output & out-dims


def sparse_mm(A_like: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Return ``A @ X`` through the vmap-safe sparse Function."""
    return _SparseMatMul.apply(A_like, X)
