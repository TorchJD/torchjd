import time
from collections.abc import Callable

import torch
from pytest import mark
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.utils._pytree import PyTree, tree_flatten, tree_map
from unit._utils import randn_
from unit.autogram._architectures import (
    Cifar10Model,
    FlatNonSequentialNN,
    ModelWithFreeParameter,
    ModelWithInterModuleParameterReuse,
    ModelWithModuleReuse,
    ModelWithNoFreeParameter,
    ModuleWithFrozenParam,
    ModuleWithParameterReuse,
    ModuleWithUnusedParam,
    PyTreeModel,
    PyTreeModule,
    ResNet18,
    ShapedModule,
    SingleInputSingleOutputModel,
    SingleInputSingleOutputModel2,
)
from unit.autojac._transform._dict_assertions import assert_tensor_dicts_are_close
from unit.conftest import DEVICE

from torchjd import backward
from torchjd._autogram._rev_gram_acc import augment_model_with_iwrm_autogram
from torchjd._autojac._transform import Diagonalize, Init, Jac, OrderedSet
from torchjd._autojac._transform._aggregate import _Matrixify
from torchjd._autojac._utils import get_leaf_tensors
from torchjd.aggregation import Aggregator, Mean, UPGrad


@mark.parametrize(
    ["architecture", "batch_size"],
    [
        (Cifar10Model, 64),
        (FlatNonSequentialNN, 64),
        (SingleInputSingleOutputModel, 64),
        (SingleInputSingleOutputModel2, 64),
        (PyTreeModel, 64),
        (ModelWithFreeParameter, 64),
        (ModelWithNoFreeParameter, 64),
        (ResNet18, 16),
    ],
)
def test_speed(architecture: type[ShapedModule], batch_size: int):
    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES
    inputs = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    model = architecture().to(device=DEVICE)

    A = Mean()
    W = A.weighting

    print(f"\nTimes for forward + backward with BS={batch_size}, A={A} on {DEVICE}.")

    def fn_autograd():
        autograd_forward_backward(model, inputs, loss_fn)

    def init_fn_autograd():
        torch.cuda.empty_cache()
        fn_autograd()

    def fn_autojac():
        autojac_forward_backward(model, inputs, loss_fn, A)

    def init_fn_autojac():
        torch.cuda.empty_cache()
        fn_autojac()

    def fn_autogram():
        autogram_forward_backward(model, inputs, loss_fn)

    def init_fn_autogram():
        torch.cuda.empty_cache()
        fn_autogram()

    def optionally_cuda_sync():
        if str(DEVICE).startswith("cuda"):
            torch.cuda.synchronize()

    def pre_fn():
        model.zero_grad()
        optionally_cuda_sync()

    def post_fn():
        optionally_cuda_sync()

    n_runs = 10
    autograd_times = torch.tensor(time_call(fn_autograd, init_fn_autograd, pre_fn, post_fn, n_runs))
    print(f"autograd times (avg = {autograd_times.mean():.5f}, std = {autograd_times.std():.5f}")
    print(autograd_times)
    print()

    autojac_times = torch.tensor(time_call(fn_autojac, init_fn_autojac, pre_fn, post_fn, n_runs))
    print(f"autojac times (avg = {autojac_times.mean():.5f}, std = {autojac_times.std():.5f}")
    print(autojac_times)
    print()

    handle = augment_model_with_iwrm_autogram(model, W)
    autogram_times = torch.tensor(time_call(fn_autogram, init_fn_autogram, pre_fn, post_fn, n_runs))
    handle.remove()
    print(f"autogram times (avg = {autogram_times.mean():.5f}, std = {autogram_times.std():.5f}")
    print(autogram_times)
    print()


@mark.parametrize(
    ["architecture", "batch_size", "n_iter"],
    [
        (Cifar10Model, 64, 1),
        (FlatNonSequentialNN, 64, 5),
        (SingleInputSingleOutputModel, 64, 5),
        (SingleInputSingleOutputModel2, 64, 5),
        (PyTreeModel, 64, 5),
        (ModuleWithParameterReuse, 64, 5),
        (ModelWithInterModuleParameterReuse, 64, 5),
        (ModelWithModuleReuse, 64, 5),
        (ModelWithFreeParameter, 64, 5),
        (ModelWithNoFreeParameter, 64, 5),
        (ModuleWithUnusedParam, 64, 5),
        (ModuleWithFrozenParam, 64, 5),
        (ResNet18, 16, 1),
        (PyTreeModule, 32, 5),
    ],
)
def test_equivalence(architecture: type[ShapedModule], batch_size: int, n_iter: int):
    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    A = UPGrad()
    W = A.weighting.weighting

    torch.manual_seed(0)
    model_autojac = architecture().to(device=DEVICE)
    torch.manual_seed(0)
    model_autogram = architecture().to(device=DEVICE)

    augment_model_with_iwrm_autogram(model_autogram, W)
    optimizer_autojac = SGD(model_autojac.parameters())
    optimizer_autogram = SGD(model_autogram.parameters())

    for i in range(n_iter):
        inputs = make_tensors(batch_size, input_shapes)
        targets = make_tensors(batch_size, output_shapes)
        loss_fn = make_mse_loss_fn(targets)

        autojac_forward_backward(model_autojac, inputs, loss_fn, A)
        expected_grads = {
            name: p.grad for name, p in model_autojac.named_parameters() if p.grad is not None
        }

        autogram_forward_backward(model_autogram, inputs, loss_fn)
        grads = {
            name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None
        }

        assert_tensor_dicts_are_close(grads, expected_grads)

        optimizer_autojac.step()
        model_autojac.zero_grad()

        optimizer_autogram.step()
        model_autogram.zero_grad()


@mark.parametrize(
    ["architecture", "batch_size"],
    [
        (Cifar10Model, 64),
        (FlatNonSequentialNN, 64),
        (SingleInputSingleOutputModel, 64),
        (SingleInputSingleOutputModel2, 64),
        (PyTreeModel, 64),
        (ModuleWithParameterReuse, 64),
        (ModelWithInterModuleParameterReuse, 64),
        (ModelWithModuleReuse, 64),
        (ModelWithFreeParameter, 64),
        (ModelWithNoFreeParameter, 64),
        (ModuleWithUnusedParam, 64),
        (ModuleWithFrozenParam, 64),
        (ResNet18, 16),
        (PyTreeModule, 32),
    ],
)
def test_augment_deaugment_reaugment(architecture: type[ShapedModule], batch_size: int):
    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    A = UPGrad()
    W = A.weighting.weighting
    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.manual_seed(0)
    model = architecture().to(device=DEVICE)

    autojac_forward_backward(model, input, loss_fn, A)
    autojac_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad()

    autograd_forward_backward(model, input, loss_fn)
    autograd_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }

    torch.manual_seed(0)
    model_autogram = architecture().to(device=DEVICE)

    # Augment and verify that we're equivalent to autojac
    handle = augment_model_with_iwrm_autogram(model_autogram, W)
    autogram_forward_backward(model_autogram, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autojac_grads)
    model_autogram.zero_grad()

    # Deaugment and verify that we're equivalent to autograd
    handle.remove()  # De-augment model
    autogram_forward_backward(model_autogram, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autograd_grads)
    model_autogram.zero_grad()

    # Re-augment and verify that we're equivalent to autojac
    augment_model_with_iwrm_autogram(model_autogram, W)
    autogram_forward_backward(model_autogram, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autojac_grads)


def noop():
    pass


def time_call(fn, init_fn=noop, pre_fn=noop, post_fn=noop, n_runs: int = 10) -> list[float]:
    init_fn()

    times = []
    for _ in range(n_runs):
        pre_fn()
        start = time.perf_counter()
        fn()
        post_fn()
        elapsed_time = time.perf_counter() - start
        times.append(elapsed_time)

    return times


def make_tensors(batch_size: int, tensor_shapes: PyTree) -> PyTree:
    def is_leaf(s):
        return isinstance(s, tuple) and all([isinstance(e, int) for e in s])

    return tree_map(lambda s: randn_((batch_size,) + s), tensor_shapes, is_leaf=is_leaf)


def autograd_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], Tensor],
) -> None:
    losses = forward_pass(model, inputs, loss_fn)
    losses.sum().backward()


def autojac_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], Tensor],
    aggregator: Aggregator,
) -> None:
    losses = forward_pass(model, inputs, loss_fn)
    backward(losses, aggregator=aggregator)


def autogram_forward_backward(
    model: nn.Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], Tensor],
) -> None:
    losses = forward_pass(model, inputs, loss_fn)
    losses.backward(torch.ones_like(losses))


def autojac_get_gramian(
    model: nn.Module, input: Tensor, loss_fn: Callable[[PyTree], Tensor]
) -> Tensor:
    losses = OrderedSet(forward_pass(model, input, loss_fn))

    # Transform that creates gradient outputs containing only ones.
    init = Init(losses)

    # Transform that turns the gradients into Jacobians.
    diag = Diagonalize(losses)

    # Transform that computes the required Jacobians.
    inputs = get_leaf_tensors(tensors=losses, excluded=set())
    jac = Jac(losses, inputs, None, False)

    mat = _Matrixify()

    transform = mat << jac << diag << init

    jacobian_matrices = transform({})

    gramian = sum([J @ J.T for J in jacobian_matrices.values()])
    return gramian


def forward_pass(model: nn.Module, inputs: PyTree, loss_fn: Callable[[PyTree], Tensor]) -> PyTree:
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
                mse_loss(output, target, reduction="none").flatten(start_dim=1)
                for output, target in zip(flat_outputs, flat_targets)
            ],
            dim=1,
        ).mean(dim=1)
        return losses

    return mse_loss_fn
