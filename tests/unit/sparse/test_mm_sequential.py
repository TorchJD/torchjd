import torch
from torchjd._autojac import backward
from torchjd.aggregation import UPGrad
from torchjd.sparse import sparse_mm

def test_sequential_backward():
    A = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1.0, 1.0]).coalesce()
    p = torch.tensor([1.0, 2.0], requires_grad=True)

    # Make y1 require A@p, y2 a simple L2 term
    y1 = sparse_mm(A, p.unsqueeze(1)).sum()  # shape (2,1) → scalar
    y2 = (p**2).sum()

    # Force sequential JD (no vmap) to touch the else-branch in backward()
    backward([y1, y2], UPGrad(), parallel_chunk_size=1)

    # Gradient shape & basic sanity check
    assert p.grad.shape == p.shape
    assert torch.isfinite(p.grad).all()
