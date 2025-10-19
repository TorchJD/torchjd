from collections.abc import Callable

import torch
from torch import Tensor, nn, vmap
from torch.nn.functional import mse_loss
from torch.utils._pytree import PyTree, tree_flatten, tree_map
from utils.architectures import get_in_out_shapes
from utils.contexts import fork_rng

from torchjd.aggregation import Aggregator, Weighting
from torchjd.autogram import Engine
from torchjd.autojac import backward


def autograd_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
) -> None:
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    losses.sum().backward()


def autojac_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
    aggregator: Aggregator,
) -> None:
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    backward(losses, aggregator=aggregator)


def autograd_gramian_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    params: list[nn.Parameter],
    loss_fn: Callable[[PyTree], list[Tensor]],
    weighting: Weighting,
) -> None:
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    gramian = compute_gramian_with_autograd(losses, params, retain_graph=True)
    losses.backward(weighting(gramian))


def autogram_forward_backward(
    model: nn.Module,
    engine: Engine,
    weighting: Weighting,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
) -> None:
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    gramian = engine.compute_gramian(losses)
    losses.backward(weighting(gramian))


def forward_pass(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
    reduction: Callable[[list[Tensor]], Tensor],
) -> PyTree:
    with fork_rng(seed=0):
        output = model(inputs)

    _, expected_output_shapes = get_in_out_shapes(model)
    assert tree_map(lambda t: t.shape[1:], output) == expected_output_shapes

    loss_tensors = loss_fn(output)
    losses = reduction(loss_tensors)
    return losses


def make_mse_loss_fn(targets: PyTree) -> Callable[[PyTree], list[Tensor]]:
    def mse_loss_fn(outputs: PyTree) -> list[Tensor]:
        flat_outputs, _ = tree_flatten(outputs)
        flat_targets, _ = tree_flatten(targets)

        loss_tensors = [
            mse_loss(output, target, reduction="none")
            for output, target in zip(flat_outputs, flat_targets)
        ]

        return loss_tensors

    return mse_loss_fn


def reduce_to_first_tensor(loss_tensors: list[Tensor]) -> Tensor:
    return loss_tensors[0]


def reduce_to_matrix(loss_tensors: list[Tensor]) -> Tensor:
    return torch.concat([reshape_raw_losses(t) for t in loss_tensors], dim=1)


def reduce_to_vector(loss_tensors: list[Tensor]) -> Tensor:
    return reduce_to_matrix(loss_tensors).mean(dim=1)


def reduce_to_scalar(loss_tensors: list[Tensor]) -> Tensor:
    return reduce_to_matrix(loss_tensors).mean()


def reshape_raw_losses(raw_losses: Tensor) -> Tensor:
    assert raw_losses.ndim > 0

    if raw_losses.ndim == 1:
        return raw_losses.unsqueeze(1)
    else:
        return raw_losses.flatten(start_dim=1)


def compute_gramian_with_autograd(
    output: Tensor, inputs: list[nn.Parameter], retain_graph: bool = False
) -> Tensor:
    """
    Computes the Gramian of the Jacobian of the outputs with respect to the inputs using vmapped
    calls to the autograd engine.
    """

    filtered_inputs = [input for input in inputs if input.requires_grad]

    def get_vjp(grad_outputs: Tensor) -> list[Tensor]:
        grads = torch.autograd.grad(
            output,
            filtered_inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return [grad for grad in grads if grad is not None]

    jacobians = vmap(get_vjp)(torch.diag(torch.ones_like(output)))
    jacobian_matrices = [jacobian.reshape([jacobian.shape[0], -1]) for jacobian in jacobians]
    gramian = sum([jacobian @ jacobian.T for jacobian in jacobian_matrices])

    return gramian


def compute_gramian(matrix: Tensor) -> Tensor:
    """Contracts the last dimension of matrix to make it into a Gramian."""

    indices = list(range(matrix.ndim))
    transposed_matrix = matrix.movedim(indices, indices[::-1])
    return torch.tensordot(matrix, transposed_matrix, dims=([-1], [0]))
