import torch
from torchjd.sparse import sparse_mm

def test_forward_backward_3d():
    # sparse 2×2 matrix
    A = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1.0, 1.0]).coalesce()

    # 3-D dense tensor (B=3, N=2, d=4)
    X = torch.randn(3, 2, 4, requires_grad=True)

    Y = sparse_mm(A, X)          # exercises 3-D forward branch
    loss = Y.sum()
    loss.backward()              # exercises 3-D backward branch

    # Gradient should be ones because A.T @ 1 = [1,1] → broadcast
    assert torch.allclose(X.grad, torch.ones_like(X), atol=1e-6)
