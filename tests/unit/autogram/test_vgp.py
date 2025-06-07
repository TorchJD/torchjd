import torch
from torch import Tensor
from torch.testing import assert_close

from torchjd.autogram import vgp


def test_vgp():
    m = 5
    x = torch.randn([m], requires_grad=True)

    def f(x: Tensor) -> Tensor:
        return x.sin()

    (y, vgp_fn) = vgp(f, x)

    gramian = torch.zeros([m, m])
    for i, e in enumerate(torch.eye(m)):
        gramian[i] = vgp_fn(e)

    grad = torch.autograd.grad(y, x, torch.ones_like(y))[0]
    expected_gramian = (grad**2).diag()

    assert_close(gramian, expected_gramian)
