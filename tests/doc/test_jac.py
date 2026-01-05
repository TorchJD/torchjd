"""
This file contains the test of the jac usage example, with a verification of the value of the obtained jacobians tuple.
"""

from torch.testing import assert_close


def test_jac():
    import torch

    from torchjd.autojac import jac

    param = torch.tensor([1.0, 2.0], requires_grad=True)
    # Compute arbitrary quantities that are function of param
    y1 = torch.tensor([-1.0, 1.0]) @ param
    y2 = (param**2).sum()
    jacobians = jac([y1, y2], [param])

    assert len(jacobians) == 1
    assert_close(jacobians[0], torch.tensor([[-1.0, 1.0], [2.0, 4.0]]), rtol=0.0, atol=1e-04)
