import torch
from torch.testing import assert_close


def test_mean():
    from torch import tensor

    from torchjd.aggregation import MeanWeighting, WeightedAggregator

    W = MeanWeighting()
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([1.0, 1.0, 1.0]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.5000, 0.5000]), rtol=0, atol=1e-4)


def test_sum():
    from torch import tensor

    from torchjd.aggregation import SumWeighting, WeightedAggregator

    W = SumWeighting()
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([2.0, 2.0, 2.0]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([1.0, 1.0]), rtol=0, atol=1e-4)


def test_constant():
    from torch import tensor

    from torchjd.aggregation import ConstantWeighting, WeightedAggregator

    W = ConstantWeighting(tensor([1.0, 2.0]))
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([8.0, 3.0, 3.0]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([1.0, 2.0]), rtol=0, atol=1e-4)


def test_upgrad():
    from torch import tensor

    from torchjd.aggregation import MeanWeighting, UPGradWrapper, WeightedAggregator

    W = UPGradWrapper(MeanWeighting())
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.2929, 1.9004, 1.9004]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([1.1109, 0.7894]), rtol=0, atol=1e-4)


def test_mgda():
    from torch import tensor

    from torchjd.aggregation import MGDAWeighting, WeightedAggregator

    W = MGDAWeighting()
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([1.1921e-07, 1.0000e00, 1.0000e00]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.6000, 0.4000]), rtol=0, atol=1e-4)


def test_dualproj():
    from torch import tensor

    from torchjd.aggregation import DualProjWrapper, MeanWeighting, WeightedAggregator

    W = DualProjWrapper(MeanWeighting())
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.5563, 1.1109, 1.1109]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.6109, 0.5000]), rtol=0, atol=1e-4)


def test_pcgrad():
    from torch import tensor

    from torchjd.aggregation import PCGradWeighting, WeightedAggregator

    W = PCGradWeighting()
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.5848, 3.8012, 3.8012]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([2.2222, 1.5789]), rtol=0, atol=1e-4)


def test_graddrop():
    from torch import tensor

    from torchjd.aggregation import GradDrop

    _ = torch.manual_seed(0)

    A = GradDrop()
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([6.0, 2.0, 2.0]), rtol=0, atol=1e-4)


def test_imtl_g():
    from torch import tensor

    from torchjd.aggregation import IMTLGWeighting, WeightedAggregator

    W = IMTLGWeighting()
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.0767, 1.0000, 1.0000]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.5923, 0.4077]), rtol=0, atol=1e-4)


def test_cagrad():
    import warnings

    warnings.filterwarnings("ignore")

    from torch import tensor

    from torchjd.aggregation import CAGradWeighting, WeightedAggregator

    W = CAGradWeighting(c=0.5)
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.1835, 1.2041, 1.2041]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.7041, 0.5000]), rtol=0, atol=1e-4)


def test_random():
    from torch import tensor

    from torchjd.aggregation import RandomWeighting, WeightedAggregator

    W = RandomWeighting()
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    _ = torch.manual_seed(0)
    assert_close(A(J), tensor([-2.6229, 1.0000, 1.0000]), rtol=0, atol=1e-4)
    _ = torch.manual_seed(0)
    assert_close(W(J), tensor([0.8623, 0.1377]), rtol=0, atol=1e-4)


def test_nash_mtl():
    import warnings

    warnings.filterwarnings("ignore")

    from torch import tensor

    from torchjd.aggregation import NashMTLWeighting, WeightedAggregator

    W = NashMTLWeighting(n_tasks=2)
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.0542, 0.7061, 0.7061]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.4182, 0.2878]), rtol=0, atol=1e-4)


def test_aligned_mtl():
    from torch import tensor

    from torchjd.aggregation import AlignedMTLWrapper, MeanWeighting, WeightedAggregator

    W = AlignedMTLWrapper(MeanWeighting())
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([0.2133, 0.9673, 0.9673]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.5591, 0.4083]), rtol=0, atol=1e-4)


def test_krum():
    from torch import tensor

    from torchjd.aggregation import KrumWeighting, WeightedAggregator

    W = KrumWeighting(n_byzantine=1, n_selected=4)
    A = WeightedAggregator(W)
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
    assert_close(W(J), tensor([0.2500, 0.2500, 0.0000, 0.2500, 0.2500]), rtol=0, atol=1e-4)


def test_trimmed_mean():
    from torch import tensor

    from torchjd.aggregation import TrimmedMean

    A = TrimmedMean(trim_number=1)
    J = tensor(
        [
            [1e11, 3],
            [1, -1e11],
            [-1e10, 1e10],
            [2, 2],
        ]
    )

    assert_close(A(J), tensor([1.5000, 2.5000]), rtol=0, atol=1e-4)


def test_normalizing():
    from torch import tensor

    from torchjd.aggregation import ConstantWeighting, NormalizingWrapper, WeightedAggregator

    W = NormalizingWrapper(ConstantWeighting(tensor([1.0, 2.0])), norm_p=1.0, norm_value=1.0)
    A = WeightedAggregator(W)
    J = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])

    assert_close(A(J), tensor([2.6667, 1.0000, 1.0000]), rtol=0, atol=1e-4)
    assert_close(W(J), tensor([0.3333, 0.6667]), rtol=0, atol=1e-4)
