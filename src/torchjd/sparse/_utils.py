"""Utility helpers shared by the sparse sub-package."""

from __future__ import annotations

from typing import Any

import torch

try:
    import importlib

    torch_sparse = importlib.import_module("torch_sparse")  # type: ignore
except (ModuleNotFoundError, OSError):  # pragma: no cover
    torch_sparse = None


def to_coalesced_coo(x: Any) -> torch.Tensor:
    """Convert *x* to a **coalesced** PyTorch sparse COO tensor."""

    if isinstance(x, torch.Tensor) and x.is_sparse:
        return x.coalesce()

    if torch_sparse and isinstance(x, torch_sparse.SparseTensor):  # type: ignore
        return x.to_torch_sparse_coo_tensor().coalesce()

    try:
        import scipy.sparse as sp  # pragma: no cover

        if isinstance(x, sp.spmatrix):
            coo = x.tocoo()
            indices = torch.as_tensor([coo.row, coo.col], dtype=torch.long)
            values = torch.as_tensor(coo.data, dtype=torch.get_default_dtype())
            return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
    except ModuleNotFoundError:  # pragma: no cover
        pass

    raise TypeError(f"Unsupported sparse type: {type(x)}")  # pragma: no cover
