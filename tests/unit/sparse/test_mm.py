import pytest
import torch

from torchjd.sparse import sparse_mm
from torchjd.sparse._utils import to_coalesced_coo

try:
    import importlib
    import types

    torch_sparse = importlib.import_module("torch_sparse")  # noqa: E402
    HAVE_TORCH_SPARSE = isinstance(torch_sparse, types.ModuleType)
except (ModuleNotFoundError, OSError):
    HAVE_TORCH_SPARSE = False


try:
    import scipy.sparse as sp

    HAVE_SCIPY = True
except ModuleNotFoundError:
    HAVE_SCIPY = False


def _dense_graph():
    idx = torch.tensor([[0, 1], [1, 0]])
    return torch.sparse_coo_tensor(idx, torch.ones(2)).coalesce()


def _batched_features(device):
    # shape (B, N, d) with B=3, N=2, d=4
    return torch.randn(3, 2, 4, device=device, dtype=torch.float32)


@pytest.mark.parametrize("device", ["cpu"])
def test_vmap_branch(device):
    A = _dense_graph().to(device)
    X = _batched_features(device)
    Y = sparse_mm(A, X)  # calls vmap-aware branch
    assert Y.shape == X.shape


@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy not installed")
def test_scipy_path():
    import numpy as np
    import scipy.sparse as sp

    coo = sp.coo_matrix(([1, 1], ([0, 1], [1, 0])), shape=(2, 2))
    A = to_coalesced_coo(coo)
    assert A.is_sparse and A.is_coalesced()


@pytest.mark.skipif(not HAVE_TORCH_SPARSE, reason="torch_sparse not installed")
def test_torch_sparse_path():
    import torch_sparse as tsp

    row = torch.tensor([0, 1])
    col = torch.tensor([1, 0])
    val = torch.ones(2)
    A_ts = tsp.SparseTensor(row=row, col=col, value=val, sparse_sizes=(2, 2))
    A = to_coalesced_coo(A_ts)
    X = torch.randn(2, 3)
    Y = sparse_mm(A, X)
    assert Y.shape == (2, 3)
