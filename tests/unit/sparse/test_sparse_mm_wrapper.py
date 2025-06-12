import torch
from torchjd.sparse._patch import enable_seamless_sparse

def test_torch_sparse_mm_wrapper():
    enable_seamless_sparse()             # idempotent
    A = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1., 1.]).coalesce()
    X = torch.randn(2, 3)

    out = torch.sparse.mm(A, X)          # routed through wrapper
    ref = A.to_dense() @ X
    assert torch.allclose(out, ref, atol=1e-6)
