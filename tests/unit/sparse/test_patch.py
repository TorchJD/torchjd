import torch
from torchjd.sparse._patch import enable_seamless_sparse

def test_monkey_patch_matmul():
    enable_seamless_sparse()  # idempotent
    A = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1.0, 1.0]).coalesce()
    X = torch.randn(2, 3)
    Y1 = A @ X                 # should hit sparse_mm via patched __matmul__
    Y2 = torch.tensor([[0., 0., 0.], [0., 0., 0.]])  # placeholder
    assert torch.allclose(Y1.sum(), (A.to_dense() @ X).sum())
