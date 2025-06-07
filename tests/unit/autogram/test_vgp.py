import torch
from torch import Tensor
from torch.testing import assert_close

from torchjd.autogram import vgp


def test_vgp():
    m = 2
    x = torch.tensor([1.0, 2.0, 3.0])

    def f(x: Tensor) -> Tensor:
        return torch.concatenate([x.sum().unsqueeze(0), (x**2).sum().unsqueeze(0)])

    (y, vgp_fn) = vgp(f, x)

    columns = []
    for i, e in enumerate(torch.eye(m)):
        columns.append(vgp_fn(e))

    gramian = torch.vstack(columns)

    expected_jacobian = torch.tensor([[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
    expected_gramian = expected_jacobian @ expected_jacobian.T

    assert_close(gramian, expected_gramian)
