from collections.abc import Callable

import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad
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
    model_ = extend(model)
    criterion_ = extend(criterion)
    model_output, losses, gramian = get_output_loss_and_gramian_supervised_iwrm_sequential(
        model_, criterion_, input, target
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
            if i > 0:
                with backpack(BatchGrad()):
                    output.backward(gradient=grad, inputs=input, retain_graph=True)

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
