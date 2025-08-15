from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.utils._pytree import PyTree, tree_flatten, tree_map

from torchjd.aggregation import Aggregator
from torchjd.autojac import backward


def autograd_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], Tensor],
) -> None:
    losses = _forward_pass(model, inputs, loss_fn)
    losses.sum().backward()


def autojac_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], Tensor],
    aggregator: Aggregator,
) -> None:
    losses = _forward_pass(model, inputs, loss_fn)
    backward(losses, aggregator=aggregator)


def autogram_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], Tensor],
) -> None:
    losses = _forward_pass(model, inputs, loss_fn)
    losses.backward(torch.ones_like(losses))


def _forward_pass(model: nn.Module, inputs: PyTree, loss_fn: Callable[[PyTree], Tensor]) -> PyTree:
    output = model(inputs)

    assert tree_map(lambda t: t.shape[1:], output) == model.OUTPUT_SHAPES

    losses = loss_fn(output)
    return losses


def make_mse_loss_fn(targets: PyTree) -> Callable[[PyTree], Tensor]:
    def mse_loss_fn(outputs: PyTree) -> Tensor:
        flat_outputs, _ = tree_flatten(outputs)
        flat_targets, _ = tree_flatten(targets)

        # For each (output_i, target_i) pair, compute the MSE at each coordinate and store it in
        # a matrix of shape [batch_size, dim_i], where dim_i is the number of elements of
        # output_i and target_i. Concatenate them along dim=1 to obtain a matrix of MSEs of
        # shape [batch_size, dim], where dim is the total number of elements of the outputs.
        # Then, reduce this into a vector of losses of size [batch_size], by applying the mean
        # along dim=1.
        losses = torch.concatenate(
            [
                reshape_raw_losses(mse_loss(output, target, reduction="none"))
                for output, target in zip(flat_outputs, flat_targets)
            ],
            dim=1,
        ).mean(dim=1)
        return losses

    return mse_loss_fn


def reshape_raw_losses(raw_losses: Tensor) -> Tensor:
    assert raw_losses.ndim > 0

    if raw_losses.ndim == 1:
        return raw_losses.unsqueeze(1)
    else:
        return raw_losses.flatten(start_dim=1)
