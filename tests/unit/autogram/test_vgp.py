import torch
from torch import Tensor
from torch.testing import assert_close

from torchjd.autogram._vgp import get_output_and_gramian


def test_vgp():
    x = torch.tensor([1.0, 2.0, 3.0])

    def f(x: Tensor) -> Tensor:
        return torch.concatenate([x.sum().unsqueeze(0), (x**2).sum().unsqueeze(0)])

    y, gramian = get_output_and_gramian(f, x)

    expected_jacobian = torch.tensor([[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
    expected_gramian = expected_jacobian @ expected_jacobian.T

    assert_close(gramian, expected_gramian)
