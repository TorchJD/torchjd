from collections import OrderedDict
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
    model_output, losses, gramian = get_output_loss_and_gramian_supervised_iwrm_sequential_v2(
        model, criterion, input, target
    )
    # weights = weighting(gramian)
    # weighted_loss = losses @ weights
    # weighted_loss.backward()

    return model_output, losses, 0


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


def get_output_loss_and_gramian_supervised_iwrm_sequential_v2(
    model: nn.Module, criterion: Callable, input: Tensor, target: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    # TODO

    batch_size = input.shape[0]
    gramian = torch.zeros((batch_size, batch_size), device=input.device, dtype=input.dtype)

    for layer in model.modules():
        params = list(layer.parameters(recurse=False))
        if len(params) > 0:
            add_one_time_autogram_hook(layer, gramian)

    output = model(input)
    losses = criterion(output, target)
    disable_requires_grad(model)
    enable_requires_grad(model)
    losses.sum().backward(retain_graph=True)

    # print("MODEL PARAM GRAD", next(model.parameters()).grad)
    #
    # print(gramian)

    return output, losses, gramian


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


def add_one_time_autogram_hook(module: nn.Module, global_gramian: Tensor) -> None:
    """
    Modifies in-place a module to register a pre-backward hook computing and accumulating the
    Gramian wrt its parameters.

    This hook is automatically disabled after one call.
    """

    params = list(module.parameters())
    if len(params) > 0:

        def forward_hook(module_: nn.Module, args, _):
            # Save the inputs for later use during the first backward pass
            module_._autogram_saved_args = args

        handle_fwd = module.register_forward_hook(forward_hook)

        def autogram_hook(module_: nn.Module, grad_output: Tensor) -> Tensor:
            input = module_._autogram_saved_args[
                0
            ]  # TODO: make this more general to handle the case of multiple inputs
            del module_._autogram_saved_args
            handle_fwd.remove()
            module_._backward_hooks = OrderedDict()  # TODO: kind of a hack...
            module_._backward_pre_hooks = OrderedDict()

            def get_vjp(input_j, grad_output_j) -> tuple[Tensor, ...]:
                # Note: we use unsqueeze(0) to turn a single activation (or grad_output) into a
                # "batch" of 1 activation (or grad_output). This is because some layers (e.g.
                # nn.Flatten) do not work equivalently if they're provided with a batch or with an
                # element of a batch. We thus always provide them with batches, just of a different
                # size.
                return _vjp_from_module(module_, input_j.unsqueeze(0))(
                    grad_output_j[0].unsqueeze(0)
                )  # TODO: not sure why we need the [0] here

            jacobians = torch.vmap(get_vjp)(input, grad_output)

            assert len(jacobians) == 1

            for jacobian in jacobians[0].values():
                J = jacobian.reshape((grad_output[0].shape[0], -1))
                global_gramian.add_(J @ J.T)  # Accumulate into the global gramian

            return grad_output  # no-op in terms of the normal differentiation flow

        handle_bck = module.register_full_backward_pre_hook(autogram_hook)

        def autoremove_hook(_, __, ___) -> None:
            # Note that removing a handle that was already removed seems to just be a no-op, so it's
            # not required to check that handle wasn't already removed here, or to try/except this.
            handle_bck.remove()

        module.register_full_backward_hook(autoremove_hook)


def disable_requires_grad(module: nn.Module):
    # TODO: verify that this disables completely the differentiation wrt the model's params, and
    #  that it doesn't just prevent storing .grad.
    for param in module.parameters():
        param.requires_grad_(False)


def enable_requires_grad(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(True)
