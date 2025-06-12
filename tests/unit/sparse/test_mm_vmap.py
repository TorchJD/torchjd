import torch
from torch.func import vmap

from torchjd.sparse import sparse_mm


def test_batched_vmap_forward_backward():
    """
    Touch the custom vmap rule in _SparseMatMul to push per-file coverage
    above the 90 % guideline.
    """
    A = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1.0, 1.0]).coalesce()
    B, N, d = 4, 2, 3
    X = torch.randn(B, N, d, requires_grad=True)

    # vmap over the first dim (B) so SparseMatMul.vmap executes
    def _single(inp):
        return sparse_mm(A, inp).sum()

    loss = vmap(_single)(X).sum()
    loss.backward()

    # Analytic gradient: A.T @ 1 = [1,1] broadcast to (B,N,d)
    expected = torch.ones_like(X)
    assert torch.allclose(X.grad, expected, atol=1e-6)
