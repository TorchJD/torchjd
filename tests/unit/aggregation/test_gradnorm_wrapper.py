import pytest
import torch
from pytest import mark
from torch.testing import assert_close

from torchjd.aggregation import MGDA, Sum
from torchjd.aggregation.gradnorm_wrapper import GradNormWrapper

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda"))


@mark.parametrize("device", devices)
@mark.parametrize(
    "wrapper_fn", [lambda device: GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)]
)
def test_forward_returns_scalar(wrapper_fn, device):
    """Test that forward() returns a scalar aggregated loss on a given device"""
    wrapper = wrapper_fn(device)
    loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2
    loss2 = (torch.randn(10, device=device, requires_grad=True).mean()) ** 2
    aggregated_loss = wrapper([loss1, loss2])
    assert isinstance(aggregated_loss, torch.Tensor)
    assert aggregated_loss.dim() == 0, "Aggregated loss should be scalar."
    assert torch.isfinite(aggregated_loss), "Aggregated loss must be finite."


@mark.parametrize("device", devices)
def test_loss_weights_normalization(device):
    """Test that the loss weights sum to the number of tasks on a given device"""
    wrapper = GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)
    weights = wrapper.loss_weights
    expected_sum = torch.tensor(2.0, dtype=weights.dtype, device=device)
    assert_close(weights.sum(), expected_sum, atol=1e-6, rtol=1e-05)


@mark.parametrize("device", devices)
def test_forward_with_incorrect_number_of_losses(device):
    """Test that forward() raises an error if the number of losses is incorrect"""
    loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2
    wrapper = GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)
    with pytest.raises(ValueError, match="Expected 2 losses"):
        _ = wrapper([loss1])


@mark.parametrize("device", devices)
def test_reset_clears_initial_losses(device):
    """Test that reset() clears the recorded initial losses on a given device"""
    loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2
    loss2 = (torch.randn(10, device=device, requires_grad=True).mean()) ** 2
    wrapper = GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)
    _ = wrapper([loss1, loss2])
    assert wrapper.initial_losses is not None, "Initial losses should be recorded."
    wrapper.reset()
    assert wrapper.initial_losses is None, "reset() should clear the recorded initial losses."


@mark.parametrize("device", devices)
@mark.parametrize(
    "wrapper_fn", [lambda device: GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)]
)
def test_balance_updates_loss_scale(wrapper_fn, device):
    """
    Test that calling balance() updates the internal loss_scale parameters
    Also, wen simulate a dependency on a shared parameter by modifying one loss
    """
    wrapper = wrapper_fn(device)
    loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2
    loss2 = (torch.randn(10, device=device, requires_grad=True).mean()) ** 2
    _ = wrapper([loss1, loss2])
    initial_loss_scale = wrapper.loss_scale.clone().detach()
    shared_param = torch.randn(10, device=device, requires_grad=True)
    loss1_mod = loss1 + 0.001 * shared_param.sum()
    balance_loss = wrapper.balance([loss1_mod, loss2], [shared_param])
    updated_loss_scale = wrapper.loss_scale.clone().detach()
    assert balance_loss.dim() == 0, "Balance loss should be scalar."
    assert torch.isfinite(balance_loss), "Balance loss must be finite."
    assert not torch.allclose(
        initial_loss_scale, updated_loss_scale
    ), "Loss scale parameters should be updated after balance()."


@mark.parametrize("device", devices)
@mark.parametrize(
    "wrapper_fn", [lambda device: GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)]
)
def test_balance_convergence_over_iterations(wrapper_fn, device):
    """
    Simulation of, multiple iterations of balance() to verify that the balancing loss decreases,
    suggesting convergence of the loss_scale parameters
    """
    wrapper = wrapper_fn(device)
    loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2 * 10.0
    loss2 = (torch.randn(10, device=device, requires_grad=True).mean()) ** 2
    _ = wrapper([loss1, loss2])
    shared_param = torch.randn(10, device=device, requires_grad=True)
    loss1_mod = loss1 + 0.001 * shared_param.sum()
    losses = [loss1_mod, loss2]
    balance_losses = []
    for _ in range(5):
        bl = wrapper.balance(losses, [shared_param])
        balance_losses.append(bl.item())
    for i in range(1, len(balance_losses)):
        assert (
            balance_losses[i] <= balance_losses[i - 1] + 1e-3
        ), f"Balancing loss did not decrease as expected: {balance_losses}"


@mark.parametrize("device", devices)
@mark.parametrize(
    "wrapper_fn", [lambda device: GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)]
)
def test_zero_gradient_behavior(wrapper_fn, device):
    """
    Test the scenario where one of the losses does not depend on the shared parameters,
    so that its gradient is zero. The balance() method should handle this gracefully.
    """
    wrapper = wrapper_fn(device)
    loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2
    loss2 = torch.tensor(2.0, device=device, requires_grad=True)
    _ = wrapper([loss1, loss2])
    shared_param = torch.randn(10, device=device, requires_grad=True)
    balance_loss = wrapper.balance([loss1, loss2], [shared_param])
    assert torch.isfinite(balance_loss), "Balance loss must be finite even with zero gradients."


@mark.parametrize("device", devices)
@mark.parametrize(
    "wrapper_fn", [lambda device: GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)]
)
@mark.parametrize("aggregator", [Sum(), MGDA()])
def test_gradnorm_wrapper_integration(wrapper_fn, aggregator, device):
    """
    Test integration of GradNormWrapper with an underlying torchjd aggregator
    Workflow:
      1. Individual losses computation
      2. GradNormWrapper utilisation for obtaining the aggregated weighted loss
      3. Individual losses are reweighted  using the wrapper's loss weights and passed to the aggregator
    """
    wrapper = wrapper_fn(device)
    loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2
    loss2 = (torch.randn(10, device=device, requires_grad=True).mean()) ** 2
    weighted_loss = wrapper([loss1, loss2])
    weighted_loss.backward(retain_graph=True)
    weights = wrapper.loss_weights
    reweighted_losses = [weights[i] * loss for i, loss in enumerate([loss1, loss2])]
    matrix = torch.stack(reweighted_losses).unsqueeze(1)  # shape: (num_tasks, 1)
    if device.type == "cuda":
        ctx = torch.use_deterministic_algorithms(False)
        if ctx is not None:
            with ctx:
                aggregated_underlying = aggregator(matrix)
        else:
            aggregated_underlying = aggregator(matrix)
    else:
        aggregated_underlying = aggregator(matrix)
    assert (
        aggregated_underlying.numel() == 1
    ), "Underlying aggregator output should be scalar (one element)."
    assert torch.isfinite(weighted_loss), "Aggregated loss must be finite."
    assert torch.isfinite(aggregated_underlying), "Underlying aggregator output must be finite."


@mark.parametrize("device", devices)
@mark.parametrize(
    "wrapper_fn", [lambda device: GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)]
)
@mark.parametrize("aggregator", [Sum()])
def test_gradnorm_wrapper_balance_integration(wrapper_fn, aggregator, device):
    """
    Test that using balance() updates internal loss_scale parameters and that the overall
    output is valid when combined with an underlying aggregator
    """
    wrapper = wrapper_fn(device)
    loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2
    loss2 = (torch.randn(10, device=device, requires_grad=True).mean()) ** 2
    _ = wrapper([loss1, loss2])
    initial_loss_scale = wrapper.loss_scale.clone().detach()
    shared_param = torch.randn(10, device=device, requires_grad=True)
    loss1_mod = loss1 + 0.001 * shared_param.sum()
    balance_loss = wrapper.balance([loss1_mod, loss2], [shared_param])
    updated_loss_scale = wrapper.loss_scale.clone().detach()
    assert balance_loss.dim() == 0, "Balance loss should be scalar."
    assert torch.isfinite(balance_loss), "Balance loss must be finite."
    assert not torch.allclose(
        initial_loss_scale, updated_loss_scale
    ), "Loss scale parameters should change after balance()."


@mark.parametrize("device", devices)
def test_representations(device):
    """Test the string representations of GradNormWrapper on a given device"""
    wrapper = GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)
    rep = repr(wrapper)
    s = str(wrapper)
    assert "GradNormWrapper" in rep, f"Expected 'GradNormWrapper' in repr, got: {rep}"
    assert "GradNormWrapper" in s, f"Expected 'GradNormWrapper' in str, got: {s}"
