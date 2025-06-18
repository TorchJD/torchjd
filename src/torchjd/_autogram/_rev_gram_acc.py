from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


def autogram_forward_backward(
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


class AutogramWrapper(nn.Module):
    def __init__(self, model: nn.Sequential):
        super().__init__()

        # TODO: it really seems we should use torch.autograd.Function to wrap a module rather than defining forward and backward hooks

        def forward_hook(TODO):
            ...
            # Override the forward pass by the forward pass from vjp, save the vjp function for the backward hook
            # I think we could save one such vjp function per batch element, and then we would only have to vmap on the grad_output
            # (rather than input and grad_output). This wouldn't work with vmap though, as we would have a list of functions...
            # Maybe we can simply just parallelize this?

        # Register a hook for each parametrized module
        def hook(module, grad_output) -> tuple[Tensor] or None:
            bs = grad_output.shape[0]

            # TODO: use the saved vjp_function, vmap it and call it appropriately. I
            jacobians = torch.vmap(get_vjp)(input, grad_output)

            return None

        handles = []
        for layer in model:
            has_trainable_params = any(p.requires_grad for p in layer.parameters())
            if has_trainable_params:
                print(f"Registering hook to {layer}")
                handle = layer.register_full_backward_pre_hook(hook)
                handles.append(handle)

        def disable_layer_hooks(_, __, ___) -> None:
            for handle in handles:
                handle.remove()

        model.register_full_backward_hook(disable_layer_hooks)

        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
