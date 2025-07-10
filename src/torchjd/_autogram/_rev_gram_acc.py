from collections import deque
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.autograd.graph import GradientEdge, get_gradient_edge
from torch.nn import Parameter

from torchjd.aggregation import UPGrad
from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


def augment_model(
    model: nn.Module,
    gramian: Tensor,
    target_edges_registry: list[GradientEdge],
    forward_hook_handles: list,
):
    for module in model.modules():
        params = list(module.parameters(recurse=False))
        if len(params) > 0:

            def forward_post_hook(module_, args, output):
                def tensor_backward_hook(grad):
                    def get_vjp(input_j, grad_output_j) -> tuple[Tensor, ...]:
                        return _vjp_from_module(module_, input_j.unsqueeze(0))(
                            grad_output_j.unsqueeze(0)
                        )

                    input = args[0]  # TODO: Here we suppose a single input tensor. Relax this.
                    jacobians = torch.vmap(get_vjp)(input, grad)

                    assert len(jacobians) == 1

                    for jacobian in jacobians[0].values():
                        J = jacobian.reshape((gramian.shape[0], -1))
                        gramian.addmm_(J, J.T)  # Accumulate the gramian

                output.register_hook(tensor_backward_hook)
                target_edges_registry.append(get_gradient_edge(output))
                return output

            forward_hook_handles.append(module.register_forward_hook(forward_post_hook))


def next_edges(edge: GradientEdge) -> list[GradientEdge]:
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]


def targets_to_leaf_targets(targets: list[GradientEdge]) -> list[GradientEdge]:
    targets = set(targets)
    nodes_to_traverse = deque((child, target) for target in targets for child in next_edges(target))

    visited = {child for child, _ in nodes_to_traverse}
    targets_graph = dict()

    while nodes_to_traverse:
        node, target = nodes_to_traverse.popleft()
        if node in targets:
            targets_graph[target] = node

        for child in next_edges(node):
            if child not in visited:
                nodes_to_traverse.append((child, target))
                visited.add(child)

    return list(targets - set(targets_graph.keys()))


def autogram_forward_backward(
    model: nn.Sequential,
    criterion: Callable,
    input: Tensor,
    target: Tensor,
    weighting: Weighting[PSDMatrix],
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = input.shape[0]
    gramian = torch.zeros((batch_size, batch_size), device=input.device)
    target_edges_registry = []
    forward_hook_handles = []
    augment_model(model, gramian, target_edges_registry, forward_hook_handles)

    output = model(input)
    losses = criterion(output, target)

    for handle in forward_hook_handles:
        handle.remove()

    leaf_targets = targets_to_leaf_targets(target_edges_registry)

    # Note: grad_outputs doesn't really matter here. The purpose of this is to compute the required
    # grad_outputs and trigger the tensor hooks with them
    _ = torch.autograd.grad(
        outputs=losses,
        inputs=leaf_targets,
        grad_outputs=torch.ones_like(losses),
        retain_graph=True,
    )

    weights = weighting(gramian)

    weighted_loss = losses @ weights
    weighted_loss.backward()

    return output, losses, weights


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
                gramian.addmm_(J, J.T)

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


def main():
    class SmartFlatten(nn.Module):
        """
        Flatten reducing inputs of shape [N, H, W, C] into [N, H * W * C] or reducing inputs of shape
        [H, W, C] into [H * W * C].
        """

        def forward(self, input):
            if input.dim() == 4:
                return torch.flatten(input, start_dim=1)
            elif input.dim() == 3:
                return torch.flatten(input)
            else:
                raise ValueError(f"Unsupported number of dimensions: {input.dim()}")

    class Cifar10Model(nn.Sequential):
        def __init__(self):
            layers = [
                nn.Conv2d(3, 32, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, groups=32),
                nn.Sequential(nn.MaxPool2d(2), nn.ReLU()),
                nn.Conv2d(64, 64, 3, groups=64),
                nn.Sequential(nn.MaxPool2d(3), nn.ReLU(), SmartFlatten()),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            ]
            super().__init__(*layers)

    batch_size = 64
    input_shape = (batch_size, 3, 32, 32)
    input = torch.randn(input_shape)
    target = torch.randint(0, 10, (batch_size,))

    model = Cifar10Model().to(device="cpu")
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    A = UPGrad()
    W = A.weighting.weighting

    autogram_forward_backward(model, criterion, input, target, W)


if __name__ == "__main__":
    main()
