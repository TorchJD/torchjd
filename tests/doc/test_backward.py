"""
This file contains the test of the backward usage example, with a verification of the value of the
obtained `.grad` field.
"""

from torch.testing import assert_close


def test_backward():
    import torch

    from torchjd import backward
    from torchjd.aggregation import UPGrad

    param = torch.tensor([1.0, 2.0], requires_grad=True)
    # Compute arbitrary quantities that are function of param
    y1 = torch.tensor([-1.0, 1.0]) @ param
    y2 = (param**2).sum()

    backward([y1, y2], UPGrad())

    assert_close(param.grad, torch.tensor([0.5000, 2.5000]), rtol=0.0, atol=1e-04)
