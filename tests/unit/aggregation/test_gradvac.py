import pytest
import torch
from torch.testing import assert_close

# For equivalence testing when target=0 (i.e. PCGrad behavior)
from torchjd.aggregation.gradvac import GradVac, _GradVacWeighting
from torchjd.aggregation.pcgrad import _PCGradWeighting

from ._property_testers import ExpectedStructureProperty


@pytest.mark.parametrize("aggregator", [GradVac(target=0.5, beta=0.02)])
class TestGradVac(ExpectedStructureProperty):
    """
    Test that GradVac satisfies the expected structure property.
    """

    pass


@pytest.mark.parametrize(
    "shape",
    [
        (2, 5),
        (2, 7),
        (2, 9),
        (2, 15),
        (2, 27),
        (2, 68),
        (2, 102),
        (2, 57),
        (2, 1200),
        (2, 11100),
    ],
)
def test_equivalence_gradvac_pcgrad(shape: tuple[int, int]):
    """
    Tests that _GradVacWeighting with target=0 (i.e. constant target 0, which yields PCGrad behavior)
    is equivalent to _PCGradWeighting for matrices of 2 rows.
    """
    torch.manual_seed(42)
    matrix = torch.randn(*shape)
    gradvac_weighting = _GradVacWeighting(target=0)
    pcgrad_weighting = _PCGradWeighting()
    result = gradvac_weighting(matrix)
    expected = pcgrad_weighting(matrix)
    assert_close(result, expected, atol=4e-04, rtol=0.0)


def test_ema_adaptive_target():
    """
    Test the EMA adaptive target update for GradVac when no constant target is provided.
    Here, input shape is (gradient_dim, num_tasks) with num_tasks=6.
    """
    torch.manual_seed(0)
    matrix = torch.randn(4, 6)  # 4-dimensional gradients, 6 tasks
    aggregator = GradVac(target=None, beta=0.2)
    weighting = aggregator.weighting

    _ = aggregator(matrix)
    assert weighting.ema is not None, "EMA should be initialized after first call"
    ema_first = weighting.ema.clone()
    _ = aggregator(matrix)

    assert not torch.allclose(ema_first, weighting.ema, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda():
    """
    Test on CUDA. For 3 tasks with 8-dimensional gradients, input shape should be (3, 8),
    so that the output is a tensor of shape (8,).
    """
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    matrix = torch.randn(3, 8, device="cuda")  # 3 tasks, 8-dim gradient
    aggregator = GradVac(target=0.4).to("cuda")
    weights = aggregator(matrix)
    assert weights.device.type == "cuda"
    assert weights.shape == (8,)


def test_repr_str():
    """
    Test that the string representations (repr and str) of GradVac are non-empty.
    """
    aggregator = GradVac(target=0.5, beta=0.01)
    rep = repr(aggregator)
    s = str(aggregator)
    assert isinstance(rep, str) and rep != ""
    assert isinstance(s, str) and s != ""


def test_consistency_with_fixed_seed():
    """
    Test that using a fixed seed produces consistent results for a constant target.
    Here, input shape is (gradient_dim, num_tasks) with 4 tasks.
    """
    torch.manual_seed(123)
    matrix = torch.randn(6, 4)
    aggregator = GradVac(target=0.0)  # constant target (PCGrad behavior)
    weights1 = aggregator(matrix)
    torch.manual_seed(123)
    weights2 = aggregator(matrix)
    assert_close(weights1, weights2, atol=1e-6, rtol=0.0)


def test_beta_zero():
    """
    Test that when beta=0, the EMA remains unchanged (i.e. stays at its initial state).
    """
    torch.manual_seed(0)
    matrix = torch.randn(6, 4)
    aggregator = GradVac(target=None, beta=0.0)
    _ = aggregator(matrix)
    # With beta=0, EMA should remain as all zeros.
    assert torch.allclose(
        aggregator.weighting.ema, torch.zeros_like(aggregator.weighting.ema), atol=1e-6
    )


def test_dtype_preservation():
    """
    Test that the aggregator preserves the dtype of the input.
    """
    matrix = torch.randn(6, 4, dtype=torch.float64)
    aggregator = GradVac(target=0.5)
    weights = aggregator(matrix)
    assert weights.dtype == torch.float64
