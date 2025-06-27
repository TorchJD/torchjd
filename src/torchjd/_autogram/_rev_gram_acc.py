from collections.abc import Callable

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

    vjpfuncs = []

    def make_vjp_call_from_module(module: nn.Module, diff_wrt_input: bool):
        def _vjp_from_module_v2(*inputs) -> Tensor:
            def functional_model_call_wrt_inputs(primals: dict[str, Parameter], *inputs) -> Tensor:
                all_state = {**primals, **dict(module.named_buffers())}
                return torch.func.functional_call(module, all_state, *inputs)

            def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
                all_state = {**primals, **dict(module.named_buffers())}
                return torch.func.functional_call(module, all_state, *inputs)

            if diff_wrt_input:
                out, vjpfunc_ = torch.func.vjp(
                    functional_model_call_wrt_inputs, dict(module.named_parameters()), *inputs
                )
            else:
                out, vjpfunc_ = torch.func.vjp(
                    functional_model_call, dict(module.named_parameters())
                )

            vjpfuncs.append(vjpfunc_)
            return out

        return _vjp_from_module_v2

    def _vjp_from_criterion(output_, target_) -> Tensor:
        def functional_model_call_wrt_inputs(output_) -> Tensor:
            all_state = {**dict(criterion.named_parameters()), **dict(criterion.named_buffers())}
            return torch.func.functional_call(criterion, all_state, args=(output_, target_))

        out, vjpfunc_ = torch.func.vjp(functional_model_call_wrt_inputs, output_)

        vjpfuncs.append(vjpfunc_)
        return out

    activation = input
    for i, layer in enumerate(model):
        diff_wrt_inputs = i > 0
        vjp_call = make_vjp_call_from_module(layer, diff_wrt_inputs)
        activation = torch.vmap(vjp_call)(activation)

    output = activation
    losses = torch.vmap(_vjp_from_criterion)(output, target)
    grad = torch.ones_like(losses)
    gramian = torch.zeros(bs, bs, device=grad.device)

    vjpfuncs = [torch.vmap(vjpfunc) for vjpfunc in vjpfuncs]

    for i, (layer, vjpfunction) in list(enumerate(zip(list(model) + [criterion], vjpfuncs)))[::-1]:
        if i == len(vjpfuncs) - 1:
            grad = vjpfunction(grad)[0]

        elif i == 0:
            jacobians = vjpfunction(grad)[0]
            for jacobian in jacobians.values():
                J = jacobian.reshape((bs, -1))
                gramian += J @ J.T  # Accumulate the gramian

        else:
            jacobians, grad = vjpfunction(grad)

            for jacobian in jacobians.values():
                J = jacobian.reshape((bs, -1))
                gramian += J @ J.T  # Accumulate the gramian

    weights = weighting(gramian)
    grad = weights

    for i, (layer, vjpfunction) in list(enumerate(zip(list(model) + [criterion], vjpfuncs)))[::-1]:
        if i == len(vjpfuncs) - 1:
            grad = vjpfunction(grad)[0]

        elif i == 0:
            jacobians = vjpfunction(grad)[0]
            params = dict(layer.named_parameters())
            for key, value in jacobians.items():
                params[key].grad = value.sum(0)

        else:
            jacobians, grad = vjpfunction(grad)
            params = dict(layer.named_parameters())
            for key, value in jacobians.items():
                params[key].grad = value.sum(0)

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
