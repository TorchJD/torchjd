import torch
from torchjd.sparse import sparse_mm

def test_single_forward_backward():
    A = torch.sparse_coo_tensor([[0,1],[1,0]], [1.,1.]).coalesce()
    X = torch.randn(2, 5, requires_grad=True)
    Y = sparse_mm(A, X)           # (2,5)
    loss = Y.sum()
    loss.backward()
    # gradient should equal A.T @ 1 = [1,1]
    assert torch.allclose(X.grad, torch.ones_like(X))
