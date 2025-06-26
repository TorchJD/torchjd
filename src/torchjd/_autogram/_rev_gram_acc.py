from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads

from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


def autogram_forward_backward(
    model: nn.Sequential,
    criterion: Callable,
    input: Tensor,
    target: Tensor,
    weighting: Weighting[PSDMatrix],
):
    input.requires_grad_(
        True
    )  # Somehow we need to do this otherwise the hook of the first layer will trigger too early
    bs = input.shape[0]
    gramian = torch.zeros((bs, bs), device=input.device, dtype=input.dtype)

    def hook(module, grad_input, grad_output) -> tuple[Tensor, ...] or None:
        # print(f"Calling hook2 on {module}")
        nonlocal gramian
        for p in module.parameters():
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                # print(f"handling param of shape {p.shape}")
                J = p.grad_sample.reshape((bs, -1))
                p.grad_sample = None  # "free" memory
                gramian += J @ J.T

    handles = []
    for layer in model:
        if len(list(layer.parameters())) > 0:
            handles.append(layer.register_full_backward_hook(hook))

    # Add a hook that removes all other hooks, so that they are one-time only
    def remove_hooks(module, grad_input, grad_output):
        for handle in handles:
            handle.remove()

    handle = model.register_full_backward_hook(remove_hooks)

    output = call_for_per_sample_grads(model)(input)
    loss = criterion(output, target).mean()
    loss.backward()

    handle.remove()

    weights = weighting(gramian)

    output = model(input)
    losses = criterion(output, target)
    loss = losses @ weights
    loss.backward()


def autogram_forward_backward_old(
    model: nn.Sequential,
    criterion: Callable,
    input: Tensor,
    target: Tensor,
    weighting: Weighting[PSDMatrix],
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Currently only works for IWRM supervised training of sequential models.
    """
    model_output, losses, gramian = get_output_loss_and_gramian_supervised_iwrm_sequential(
        model, criterion, input, target
    )
    weights = weighting(gramian)
    weighted_loss = losses @ weights
    weighted_loss.backward()

    return model_output, losses, weights


def get_output_loss_and_gramian_supervised_iwrm_sequential(
    model: nn.Sequential, criterion: Callable, input: Tensor, target: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    bs = input.shape[0]
    outputs = _compute_outputs(criterion, input, model, target)
    grad = torch.ones_like(outputs[-1])
    gramian = torch.zeros(bs, bs, device=grad.device)
    for i, (input, output, layer) in list(
        enumerate(zip(outputs[:-1], outputs[1:], list(model) + [criterion]))
    )[::-1]:
        params = list(layer.parameters())
        if len(params) > 0:

            def get_vjp(input_j, grad_output_j) -> tuple[Tensor, ...]:
                # Note: we use unsqueeze(0) to turn a single activation (or grad_output) into a
                # "batch" of 1 activation (or grad_output). This is because some layers (e.g.
                # nn.Flatten) do not work equivalently if they're provided with a batch or with an
                # element of a batch. We thus always provide them with batches, just of a different
                # size.
                return _vjp_from_module(layer, input_j.unsqueeze(0))(grad_output_j.unsqueeze(0))

            jacobians = torch.vmap(get_vjp)(input, grad)

            assert len(jacobians) == 1

            for jacobian in jacobians[0].values():
                J = jacobian.reshape((bs, -1))
                gramian += J @ J.T  # Accumulate the gramian

        if i == 0:
            break  # Don't try to differentiate with respect to the model's input
        grad = torch.autograd.grad(output, input, grad, retain_graph=True)[0]

    model_output = outputs[-2]
    losses = outputs[-1]
    return model_output, losses, gramian


def _compute_outputs(criterion, input, model: nn.Sequential, target) -> list[Tensor]:
    activations = [input]
    for layer in model:
        activation = layer(activations[-1])
        activations.append(activation)

    losses = criterion(activations[-1], target)
    return activations + [losses]


def _vjp_from_module(module: nn.Module, *inputs) -> Callable:
    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers())}
        return torch.func.functional_call(module, all_state, *inputs)

    return torch.func.vjp(functional_model_call, dict(module.named_parameters()))[1]
