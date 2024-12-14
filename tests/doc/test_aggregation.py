"""
This file contains tests for the usage examples provided in the Aggregator subclasses. Each
Aggregator's usage example, including its imports, should be copied in a function here, with the
only difference that the advertised output should be replaced by a call to `assert_close`. The
functions should be in alphabetical order.
"""

import torch
from torch.testing import assert_close


def test_aligned_mtl():
    from torch import tensor

    from torchjd.aggregation import AlignedMTL

    A = AlignedMTL()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.2133, 0.9673, 0.9673]), rtol=0, atol=1e-4)


def test_cagrad():
    # Extra ----------------------------------------------------------------------------------------
    import warnings

    warnings.filterwarnings("ignore")
    # ----------------------------------------------------------------------------------------------

    from torch import tensor

    from torchjd.aggregation import CAGrad

    A = CAGrad(c=0.5)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.1835, 1.2041, 1.2041]), rtol=0, atol=1e-4)


def test_constant():
    from torch import tensor

    from torchjd.aggregation import Constant

    A = Constant(tensor([1.0, 2.0]))
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([8.0, 3.0, 3.0]), rtol=0, atol=1e-4)


def test_dualproj():
    from torch import tensor

    from torchjd.aggregation import DualProj

    A = DualProj()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.5563, 1.1109, 1.1109]), rtol=0, atol=1e-4)


def test_graddrop():
    from torch import tensor

    from torchjd.aggregation import GradDrop

    _ = torch.manual_seed(0)

    A = GradDrop()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([6.0, 2.0, 2.0]), rtol=0, atol=1e-4)


def test_imtl_g():
    from torch import tensor

    from torchjd.aggregation import IMTLG

    A = IMTLG()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.0767, 1.0000, 1.0000]), rtol=0, atol=1e-4)


def test_krum():
    from torch import tensor

    from torchjd.aggregation import Krum

    A = Krum(n_byzantine=1, n_selected=4)
    J = tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [75.0, -666.0, 23],  # adversarial row
            [1.0, 2.0, 3.0],
            [2.0, 0.0, 1.0],
        ]
    )

    assert_close(A(J), tensor([1.2500, 0.7500, 1.5000]), rtol=0, atol=1e-4)


def test_mean():
    from torch import tensor

    from torchjd.aggregation import Mean

    A = Mean()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([1.0, 1.0, 1.0]), rtol=0, atol=1e-4)


def test_mgda():
    from torch import tensor

    from torchjd.aggregation import MGDA

    A = MGDA()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([1.1921e-07, 1.0000e00, 1.0000e00]), rtol=0, atol=1e-4)


def test_nash_mtl():
    # Extra ----------------------------------------------------------------------------------------
    import warnings

    warnings.filterwarnings("ignore")
    # ----------------------------------------------------------------------------------------------

    from torch import tensor

    from torchjd.aggregation import NashMTL

    A = NashMTL(n_tasks=2)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.0542, 0.7061, 0.7061]), rtol=0, atol=1e-4)


def test_pcgrad():
    from torch import tensor

    from torchjd.aggregation import PCGrad

    A = PCGrad()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.5848, 3.8012, 3.8012]), rtol=0, atol=1e-4)


def test_random():
    from torch import tensor

    from torchjd.aggregation import Random

    A = Random()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    # Extra ----------------------------------------------------------------------------------------
    _ = torch.manual_seed(0)
    # ----------------------------------------------------------------------------------------------

    assert_close(A(J), tensor([-2.6229, 1.0000, 1.0000]), rtol=0, atol=1e-4)
    assert_close(A(J), tensor([5.3976, 1.0000, 1.0000]), rtol=0, atol=1e-4)


def test_sum():
    from torch import tensor

    from torchjd.aggregation import Sum

    A = Sum()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([2.0, 2.0, 2.0]), rtol=0, atol=1e-4)


def test_trimmed_mean():
    from torch import tensor

    from torchjd.aggregation import TrimmedMean

    A = TrimmedMean(trim_number=1)
    J = tensor(
        [
            [1e11, 3.0],
            [1.0, -1e11],
            [-1e10, 1e10],
            [2.0, 2.0],
        ]
    )

    assert_close(A(J), tensor([1.5000, 2.5000]), rtol=0, atol=1e-4)


def test_upgrad():
    from torch import tensor

    from torchjd.aggregation import UPGrad

    A = UPGrad()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.2929, 1.9004, 1.9004]), rtol=0, atol=1e-4)
