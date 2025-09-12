from itertools import combinations

import pytest
import torch
from pytest import mark, param
from torch import nn
from torch.optim import SGD
from torch.testing import assert_close
from unit.conftest import DEVICE
from utils.architectures import (
    AlexNet,
    Cifar10Model,
    FreeParam,
    GroupNormMobileNetV3Small,
    InstanceNormMobileNetV2,
    InstanceNormResNet18,
    InterModuleParamReuse,
    MIMOBranched,
    MISOBranched,
    ModuleReuse,
    MultiInputMultiOutput,
    MultiInputSingleOutput,
    MultiOutputWithFrozenBranch,
    Ndim0Output,
    Ndim1Output,
    Ndim2Output,
    Ndim3Output,
    Ndim4Output,
    NoFreeParam,
    OverlyNested,
    PIPOBranched,
    PISOBranched,
    PyTreeInputPyTreeOutput,
    PyTreeInputSingleOutput,
    Randomness,
    ShapedModule,
    SimpleBranched,
    SimpleParamReuse,
    SingleInputPyTreeOutput,
    SIPOBranched,
    SomeFrozenParam,
    SomeUnusedOutput,
    SomeUnusedParam,
    SqueezeNet,
    WithBuffered,
    WithModuleTrackingRunningStats,
    WithNoTensorOutput,
    WithRNN,
    WithSideEffect,
    WithSomeFrozenModule,
)
from utils.autograd_compute_gramian import compute_gramian_with_autograd
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.forward_backwards import (
    autograd_forward_backward,
    autograd_gramian_forward_backward,
    autogram_forward_backward,
    make_mse_loss_fn,
)
from utils.tensors import make_tensors

from torchjd.aggregation import UPGradWeighting
from torchjd.autogram._engine import Engine

PARAMETRIZATIONS = [
    (OverlyNested, 32),
    (MultiInputSingleOutput, 32),
    (MultiInputMultiOutput, 32),
    (SingleInputPyTreeOutput, 32),
    (PyTreeInputSingleOutput, 32),
    (PyTreeInputPyTreeOutput, 32),
    (SimpleBranched, 32),
    (SimpleBranched, SimpleBranched.INPUT_SHAPES[0]),  # Edge case: batch_size = input dim
    (MIMOBranched, 32),
    (MISOBranched, 32),
    (SIPOBranched, 32),
    (PISOBranched, 32),
    (PIPOBranched, 1),
    (PIPOBranched, 2),
    (PIPOBranched, 32),
    (WithNoTensorOutput, 32),
    (WithBuffered, 32),
    (SimpleParamReuse, 32),
    (InterModuleParamReuse, 32),
    (ModuleReuse, 32),
    (SomeUnusedParam, 32),
    (SomeFrozenParam, 32),
    (MultiOutputWithFrozenBranch, 32),
    (WithSomeFrozenModule, 32),
    param(WithSideEffect, 32, marks=mark.xfail),
    (SomeUnusedOutput, 32),
    (Ndim0Output, 32),
    (Ndim1Output, 32),
    (Ndim2Output, 32),
    (Ndim3Output, 32),
    (Ndim4Output, 32),
    (FreeParam, 32),
    (NoFreeParam, 32),
    param(Randomness, 32, marks=mark.xfail),
    param(Cifar10Model, 16, marks=[mark.slow, mark.garbage_collect]),
    param(AlexNet, 2, marks=[mark.slow, mark.garbage_collect]),
    param(InstanceNormResNet18, 4, marks=[mark.slow, mark.garbage_collect]),
    param(GroupNormMobileNetV3Small, 3, marks=[mark.slow, mark.garbage_collect]),
    param(SqueezeNet, 8, marks=[mark.slow, mark.garbage_collect]),
    param(InstanceNormMobileNetV2, 2, marks=[mark.slow, mark.garbage_collect]),
]


@mark.parametrize(["architecture", "batch_size"], PARAMETRIZATIONS)
def test_gramian_equivalence_autograd_autogram(
    architecture: type[ShapedModule],
    batch_size: int,
):
    """
    Tests that the autograd and the autogram engines compute the same gramian.
    """

    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    torch.manual_seed(0)
    model_autograd = architecture().to(device=DEVICE)
    torch.manual_seed(0)
    model_autogram = architecture().to(device=DEVICE)

    engine = Engine(model_autogram.modules())

    inputs = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.random.manual_seed(0)  # Fix randomness for random aggregators and random models
    output = model_autograd(inputs)
    losses = loss_fn(output)
    autograd_gramian = compute_gramian_with_autograd(losses, list(model_autograd.parameters()))

    torch.random.manual_seed(0)  # Fix randomness for random weightings and random models
    output = model_autogram(inputs)
    losses = loss_fn(output)
    autogram_gramian = engine.compute_gramian(losses)

    assert_close(autogram_gramian, autograd_gramian)


@mark.parametrize(["architecture", "batch_size"], PARAMETRIZATIONS)
def test_equivalence_autograd_autogram(
    architecture: type[ShapedModule],
    batch_size: int,
):
    """
    Tests that the autogram engine gives the same results as the autograd engine on IWRM for several
    JD steps.
    """

    n_iter = 3

    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    weighting = UPGradWeighting()

    torch.manual_seed(0)
    model_autograd = architecture().to(device=DEVICE)
    torch.manual_seed(0)
    model_autogram = architecture().to(device=DEVICE)

    engine = Engine(model_autogram.modules())
    optimizer_autograd = SGD(model_autograd.parameters(), lr=1e-7)
    optimizer_autogram = SGD(model_autogram.parameters(), lr=1e-7)

    for i in range(n_iter):
        inputs = make_tensors(batch_size, input_shapes)
        targets = make_tensors(batch_size, output_shapes)
        loss_fn = make_mse_loss_fn(targets)

        torch.random.manual_seed(0)  # Fix randomness for random aggregators and random models
        autograd_gramian_forward_backward(
            model_autograd, inputs, list(model_autograd.parameters()), loss_fn, weighting
        )
        expected_grads = {
            name: p.grad for name, p in model_autograd.named_parameters() if p.grad is not None
        }

        torch.random.manual_seed(0)  # Fix randomness for random weightings and random models
        autogram_forward_backward(model_autogram, engine, weighting, inputs, loss_fn)
        grads = {
            name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None
        }

        assert_tensor_dicts_are_close(grads, expected_grads)

        optimizer_autograd.step()
        model_autograd.zero_grad()

        optimizer_autogram.step()
        model_autogram.zero_grad()


@mark.parametrize(["architecture", "batch_size"], PARAMETRIZATIONS)
def test_autograd_while_modules_are_hooked(architecture: type[ShapedModule], batch_size: int):
    """
    Tests that the hooks added when constructing the engine do not interfere with a simple autograd
    call.
    """

    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    W = UPGradWeighting()
    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.manual_seed(0)
    model = architecture().to(device=DEVICE)

    torch.manual_seed(0)  # Fix randomness for random models
    autograd_gramian_forward_backward(model, input, list(model.parameters()), loss_fn, W)
    autograd_gramian_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad()

    torch.manual_seed(0)  # Fix randomness for random models
    autograd_forward_backward(model, input, loss_fn)
    autograd_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }

    torch.manual_seed(0)
    model_autogram = architecture().to(device=DEVICE)

    # Hook modules and verify that we're equivalent to autograd when using the engine
    engine = Engine(model_autogram.modules())
    torch.manual_seed(0)  # Fix randomness for random models
    autogram_forward_backward(model_autogram, engine, W, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autograd_gramian_grads)
    model_autogram.zero_grad()

    # Verify that even with the hooked modules, autograd works normally when not using the engine.
    # Results should be the same as a normal call to autograd, and no time should be spent computing
    # the gramian at all.
    torch.manual_seed(0)  # Fix randomness for random models
    autograd_forward_backward(model_autogram, input, loss_fn)
    assert engine._gramian_accumulator.gramian is None
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autograd_grads)
    model_autogram.zero_grad()


def _non_empty_subsets(elements: set) -> list[set]:
    """
    Generates the list of subsets of the given set, excluding the empty set.
    """
    return [set(c) for r in range(1, len(elements) + 1) for c in combinations(elements, r)]


@mark.parametrize("gramian_module_names", _non_empty_subsets({"fc0", "fc1", "fc2", "fc3", "fc4"}))
def test_partial_autogram(gramian_module_names: set[str]):
    """
    Tests that partial JD via the autogram engine works similarly as if the gramian was computed via
    the autograd engine.

    Note that this test is a bit redundant now that we have the Engine interface, because it now
    just compares two ways of computing the Gramian, which is independant of the idea of partial JD.
    """

    architecture = SimpleBranched
    batch_size = 64

    weighting = UPGradWeighting()

    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.manual_seed(0)
    model = architecture().to(device=DEVICE)

    output = model(input)
    losses = loss_fn(output)

    gramian_modules = [model.get_submodule(name) for name in gramian_module_names]
    gramian_params = []
    for m in gramian_modules:
        gramian_params += list(m.parameters())

    gramian = compute_gramian_with_autograd(losses, gramian_params, retain_graph=True)
    torch.manual_seed(0)
    losses.backward(weighting(gramian))

    expected_grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    engine = Engine(gramian_modules)

    output = model(input)
    losses = loss_fn(output)
    gramian = engine.compute_gramian(losses)
    torch.manual_seed(0)
    losses.backward(weighting(gramian))

    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, expected_grads)


@mark.parametrize("architecture", [WithRNN, WithModuleTrackingRunningStats])
def test_incompatible_modules(architecture: type[nn.Module]):
    """Tests that the engine cannot be constructed with incompatible modules."""

    model = architecture().to(device=DEVICE)

    with pytest.raises(ValueError):
        _ = Engine(model.modules())
