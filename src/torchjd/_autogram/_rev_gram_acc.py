from collections.abc import Callable
from queue import LifoQueue
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


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
    bs = input.shape[0]

    vjpfuncs = LifoQueue()

    def make_vjp_call_from_module(module: nn.Module, diff_wrt_input: bool):
        def _vjp_from_module_v2(*inputs) -> Tuple[Tensor, Any]:
            def functional_model_call_wrt_inputs(primals: dict[str, Parameter], *inputs) -> Tensor:
                all_state = {**primals, **dict(module.named_buffers())}
                return torch.func.functional_call(module, all_state, *inputs)

            def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
                all_state = {**primals, **dict(module.named_buffers())}
                return torch.func.functional_call(module, all_state, *inputs)

            if diff_wrt_input:
                return torch.func.vjp(
                    functional_model_call_wrt_inputs, dict(module.named_parameters()), *inputs
                )
            else:
                return torch.func.vjp(functional_model_call, dict(module.named_parameters()))

        return _vjp_from_module_v2

    def _vjp_from_criterion(output_, target_) -> Tuple[Tensor, Callable]:
        def functional_model_call_wrt_inputs(output_) -> Tensor:
            all_state = {**dict(criterion.named_parameters()), **dict(criterion.named_buffers())}
            return torch.func.functional_call(criterion, all_state, args=(output_, target_))

        return torch.func.vjp(functional_model_call_wrt_inputs, output_)

    activation = input
    for i, layer in enumerate(model):
        diff_wrt_inputs = i > 0
        vjp_call = make_vjp_call_from_module(layer, diff_wrt_inputs)
        activation, vjp_func = torch.vmap(vjp_call, out_dims=(0, None))(activation)
        vjpfuncs.put(vjp_func)

    output = activation
    losses, vjp_func = torch.vmap(_vjp_from_criterion, out_dims=(0, None))(output, target)
    vjpfuncs.put(vjp_func)
    grad = torch.ones_like(losses)
    gramian = torch.zeros(bs, bs, device=grad.device)

    for i, layer in list(enumerate(list(model) + [criterion]))[::-1]:
        vjp_function = torch.vmap(vjpfuncs.get())
        if i == len(model):
            grad = vjp_function(grad)[0]

        elif i == 0:
            jacobians = vjp_function(grad)[0]
            for jacobian in jacobians.values():
                J = jacobian.reshape((bs, -1))
                gramian += J @ J.T  # Accumulate the gramian

        else:
            jacobians, grad = vjp_function(grad)

            for jacobian in jacobians.values():
                J = jacobian.reshape((bs, -1))
                gramian += J @ J.T  # Accumulate the gramian

    weights = weighting(gramian)
    losses.backward(weights)

    return output, losses, weights


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
