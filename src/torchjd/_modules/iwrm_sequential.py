from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


class IWRMSequential(Module):
    def __init__(self, weighting: Weighting[PSDMatrix], layers: list[Module]):
        super().__init__()
        self.layers = layers
        self.weighting = weighting

    def forward(self, input: Any):
        activations = self._compute_activations(input)
        output = activations[-1]

        def hook(grad: Tensor) -> Tensor:
            gramian = self._autogram(activations, output.shape[0])
            scaled_gramian = grad[:, None] * gramian * grad[None, :]
            # Note that this is different from having a pref_vector for upgrad when some element of
            # grad is 0 or negative.
            weights = self.weighting(scaled_gramian)
            return weights

        output.register_hook(hook)

        return output

    def _compute_activations(self, input) -> list[Tensor]:
        activations = [input]
        for layer in self.layers:
            activation = layer(activations[-1])
            activations.append(activation)
        return activations

    def _autogram(self, activations, batch_size):
        grad = torch.ones_like(activations[-1])
        gramian = torch.zeros(batch_size, batch_size)
        for i, (input, output, layer) in list(
            enumerate(zip(activations[:-1], activations[1:], self.layers))
        )[::-1]:
            params = list(layer.parameters())
            if len(params) > 0:

                def get_vjp(input_j, grad_output_j) -> tuple[Tensor, ...]:
                    return _vjp_from_module(layer, input_j)(grad_output_j)

                jacobians = torch.vmap(get_vjp)(input, grad)

                assert len(jacobians) == 1

                for jacobian in jacobians[0].values():
                    J = jacobian.reshape((batch_size, -1))
                    gramian += J @ J.T  # Accumulate the gramian

            if i == 0:
                break  # Don't try to differentiate with respect to the model's input
            grad = torch.autograd.grad(output, input, grad, retain_graph=False)[0]

        return gramian


def _vjp_from_module(module: Module, *inputs) -> Callable:
    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers())}
        return torch.func.functional_call(module, all_state, *inputs)

    return torch.func.vjp(functional_model_call, dict(module.named_parameters()))[1]
