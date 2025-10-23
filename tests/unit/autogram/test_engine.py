from collections.abc import Callable
from itertools import combinations
from math import prod

import torch
from pytest import mark, param
from torch import Tensor
from torch.nn import BatchNorm2d, InstanceNorm2d, Linear, Module, Parameter
from torch.optim import SGD
from torch.testing import assert_close
from torch.utils._pytree import PyTree
from utils.architectures import (
    AlexNet,
    Cifar10Model,
    FreeParam,
    GroupNormMobileNetV3Small,
    InstanceNormMobileNetV2,
    InstanceNormResNet18,
    InterModuleParamReuse,
    IntraModuleParamReuse,
    MIMOBranched,
    MISOBranched,
    ModelAlsoUsingSubmoduleParamsDirectly,
    ModelUsingSubmoduleParamsDirectly,
    ModuleFactory,
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
    RequiresGradOfSchrodinger,
    SimpleBranched,
    SingleInputPyTreeOutput,
    SIPOBranched,
    SomeFrozenParam,
    SomeUnusedOutput,
    SomeUnusedParam,
    SqueezeNet,
    WithBuffered,
    WithDropout,
    WithModuleWithHybridPyTreeArg,
    WithModuleWithHybridPyTreeKwarg,
    WithModuleWithStringArg,
    WithModuleWithStringKwarg,
    WithModuleWithStringOutput,
    WithMultiHeadAttention,
    WithNoTensorOutput,
    WithRNN,
    WithSideEffect,
    WithSomeFrozenModule,
    WithTransformer,
    WithTransformerLarge,
)
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.forward_backwards import (
    CloneParams,
    autograd_forward_backward,
    autogram_forward_backward,
    compute_gramian,
    compute_gramian_with_autograd,
    forward_pass,
    make_mse_loss_fn,
    reduce_to_first_tensor,
    reduce_to_matrix,
    reduce_to_scalar,
    reduce_to_vector,
)
from utils.tensors import make_inputs_and_targets, ones_, randn_, zeros_

from torchjd.aggregation import UPGradWeighting
from torchjd.autogram._engine import Engine
from torchjd.autogram._gramian_utils import movedim_gramian, reshape_gramian

BASE_PARAMETRIZATIONS = [
    (ModuleFactory(OverlyNested), 32),
    (ModuleFactory(MultiInputSingleOutput), 32),
    (ModuleFactory(MultiInputMultiOutput), 32),
    (ModuleFactory(SingleInputPyTreeOutput), 32),
    (ModuleFactory(PyTreeInputSingleOutput), 32),
    (ModuleFactory(PyTreeInputPyTreeOutput), 32),
    (ModuleFactory(SimpleBranched), 32),
    (ModuleFactory(SimpleBranched), SimpleBranched.INPUT_SHAPES[0]),  # Edge case: bs = input dim
    (ModuleFactory(MIMOBranched), 32),
    (ModuleFactory(MISOBranched), 32),
    (ModuleFactory(SIPOBranched), 32),
    (ModuleFactory(PISOBranched), 32),
    (ModuleFactory(PIPOBranched), 1),
    (ModuleFactory(PIPOBranched), 2),
    (ModuleFactory(PIPOBranched), 32),
    (ModuleFactory(WithNoTensorOutput), 32),
    (ModuleFactory(WithBuffered), 32),
    (ModuleFactory(IntraModuleParamReuse), 32),
    (ModuleFactory(ModuleReuse), 32),
    (ModuleFactory(SomeUnusedParam), 32),
    (ModuleFactory(SomeFrozenParam), 32),
    (ModuleFactory(MultiOutputWithFrozenBranch), 32),
    (ModuleFactory(WithSomeFrozenModule), 32),
    (ModuleFactory(RequiresGradOfSchrodinger), 32),
    (ModuleFactory(SomeUnusedOutput), 32),
    (ModuleFactory(Ndim0Output), 32),
    (ModuleFactory(Ndim1Output), 32),
    (ModuleFactory(Ndim2Output), 32),
    (ModuleFactory(Ndim3Output), 32),
    (ModuleFactory(Ndim4Output), 32),
    (ModuleFactory(WithDropout), 32),
    (ModuleFactory(WithModuleWithStringArg), 32),
    (ModuleFactory(WithModuleWithHybridPyTreeArg), 32),
    (ModuleFactory(WithModuleWithStringOutput), 32),
    (ModuleFactory(WithModuleWithStringKwarg), 32),
    (ModuleFactory(WithModuleWithHybridPyTreeKwarg), 32),
    (ModuleFactory(WithMultiHeadAttention), 32),
    param(
        ModuleFactory(WithTransformer),
        32,
        marks=mark.filterwarnings("ignore:There is a performance drop"),
    ),
    (ModuleFactory(FreeParam), 32),
    (ModuleFactory(NoFreeParam), 32),
    (ModuleFactory(Randomness), 32),
    (ModuleFactory(InstanceNorm2d, num_features=3, affine=True, track_running_stats=True), 32),
    (ModuleFactory(BatchNorm2d, num_features=3, affine=True, track_running_stats=False), 32),
    param(ModuleFactory(Cifar10Model), 16, marks=mark.slow),
    param(ModuleFactory(AlexNet), 2, marks=mark.slow),
    param(ModuleFactory(InstanceNormResNet18), 4, marks=mark.slow),
    param(ModuleFactory(GroupNormMobileNetV3Small), 3, marks=mark.slow),
    param(ModuleFactory(SqueezeNet), 8, marks=mark.slow),
    param(ModuleFactory(InstanceNormMobileNetV2), 2, marks=mark.slow),
    param(
        ModuleFactory(WithTransformerLarge),
        8,
        marks=[mark.slow, mark.filterwarnings("ignore:There is a performance drop")],
    ),
]

# These parametrizations are expected to fail on test_autograd_while_modules_are_hooked
_SPECIAL_PARAMETRIZATIONS = [
    (ModuleFactory(WithSideEffect), 32),  # When use_engine=True, double side-effect
    param(ModuleFactory(WithRNN), 32, marks=mark.xfail_if_cuda),  # Does not fail on cuda
    # when use_engine=False because engine is not even used.
]

PARAMETRIZATIONS = BASE_PARAMETRIZATIONS + _SPECIAL_PARAMETRIZATIONS


def _assert_gramian_is_equivalent_to_autograd(factory: ModuleFactory, batch_size: int):
    model_autograd, model_autogram = factory(), factory()
    engine = Engine(model_autogram)
    inputs, targets = make_inputs_and_targets(model_autograd, batch_size)
    loss_fn = make_mse_loss_fn(targets)

    losses, params = _get_losses_and_params(model_autograd, inputs, loss_fn, reduce_to_vector)
    autograd_gramian = compute_gramian_with_autograd(losses, params)

    losses = forward_pass(model_autogram, inputs, loss_fn, reduce_to_vector)
    autogram_gramian = engine.compute_gramian(losses)

    assert_close(autogram_gramian, autograd_gramian, rtol=1e-4, atol=3e-5)


def _get_losses_and_params_with_cross_terms(
    model: Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
    reduction: Callable[[list[Tensor]], Tensor],
) -> tuple[Tensor, list[Parameter]]:
    losses = forward_pass(model, inputs, loss_fn, reduction)
    params = list(model.parameters())
    return losses, params


def _get_losses_and_params_without_cross_terms(
    model: Module,
    inputs: PyTree,
    loss_fn: Callable[[PyTree], list[Tensor]],
    reduction: Callable[[list[Tensor]], Tensor],
) -> tuple[Tensor, list[Parameter]]:
    # Not considering cross-terms (except intra-module parameter reuse):
    with CloneParams(model) as params:
        losses = forward_pass(model, inputs, loss_fn, reduction)

    return losses, params


_get_losses_and_params = _get_losses_and_params_with_cross_terms


@mark.parametrize(["factory", "batch_size"], PARAMETRIZATIONS)
def test_compute_gramian(factory: ModuleFactory, batch_size: int):
    """Tests that the autograd and the autogram engines compute the same gramian."""

    _assert_gramian_is_equivalent_to_autograd(factory, batch_size)


@mark.xfail
@mark.parametrize(
    "factory",
    [
        ModuleFactory(ModelUsingSubmoduleParamsDirectly),
        ModuleFactory(ModelAlsoUsingSubmoduleParamsDirectly),
        ModuleFactory(InterModuleParamReuse),
    ],
)
@mark.parametrize("batch_size", [1, 3, 32])
def test_compute_gramian_unsupported_architectures(factory: ModuleFactory, batch_size: int):
    """
    Tests compute_gramian on some architectures that are known to be unsupported. It is expected to
    fail.
    """

    _assert_gramian_is_equivalent_to_autograd(factory, batch_size)


@mark.parametrize("batch_size", [1, 3, 16])
@mark.parametrize(
    ["reduction", "movedim_source", "movedim_destination"],
    [
        # 0D
        (reduce_to_scalar, [], []),  # ()
        # 1D
        (reduce_to_vector, [], []),  # (batch_size,)
        # 2D
        (reduce_to_matrix, [], []),  # (batch_size, d1 * d2)
        (reduce_to_matrix, [0], [1]),  # (d1 * d2, batch_size)
        # 3D
        (reduce_to_first_tensor, [], []),  # (batch_size, d1, d2)
        (reduce_to_first_tensor, [0], [1]),  # (d1, batch_size, d2)
        (reduce_to_first_tensor, [0], [2]),  # (d2, d1, batch_size)
    ],
)
def test_compute_gramian_various_output_shapes(
    batch_size: int | None,
    reduction: Callable[[list[Tensor]], Tensor],
    movedim_source: list[int],
    movedim_destination: list[int],
):
    """
    Tests that the autograd and the autogram engines compute the same gramian when the output can
    have various different shapes, and can be batched in any of its dimensions.
    """

    factory = ModuleFactory(Ndim2Output)
    model_autograd, model_autogram = factory(), factory()
    inputs, targets = make_inputs_and_targets(model_autograd, batch_size)
    loss_fn = make_mse_loss_fn(targets)

    losses, params = _get_losses_and_params(model_autograd, inputs, loss_fn, reduction)
    reshaped_losses = torch.movedim(losses, movedim_source, movedim_destination)
    # Go back to a vector so that compute_gramian_with_autograd works
    loss_vector = reshaped_losses.reshape([-1])
    autograd_gramian = compute_gramian_with_autograd(loss_vector, params)
    expected_gramian = reshape_gramian(autograd_gramian, list(reshaped_losses.shape))

    engine = Engine(model_autogram)
    losses = forward_pass(model_autogram, inputs, loss_fn, reduction)
    reshaped_losses = torch.movedim(losses, movedim_source, movedim_destination)
    autogram_gramian = engine.compute_gramian(reshaped_losses)

    assert_close(autogram_gramian, expected_gramian, rtol=1e-4, atol=1e-5)


def _non_empty_subsets(elements: set) -> list[set]:
    """
    Generates the list of subsets of the given set, excluding the empty set.
    """
    return [set(c) for r in range(1, len(elements) + 1) for c in combinations(elements, r)]


@mark.parametrize("gramian_module_names", _non_empty_subsets({"fc0", "fc1", "fc2", "fc3", "fc4"}))
def test_compute_partial_gramian(gramian_module_names: set[str]):
    """
    Tests that the autograd and the autogram engines compute the same gramian when only a subset of
    the model parameters is specified.
    """

    model = ModuleFactory(SimpleBranched)()
    batch_size = 64
    inputs, targets = make_inputs_and_targets(model, batch_size)
    loss_fn = make_mse_loss_fn(targets)
    gramian_modules = [model.get_submodule(name) for name in gramian_module_names]
    gramian_params = []
    for m in gramian_modules:
        gramian_params += list(m.parameters())

    # This includes cross-terms, but the model has no parameter reuse.
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    autograd_gramian = compute_gramian_with_autograd(losses, gramian_params, retain_graph=True)

    engine = Engine(*gramian_modules)
    losses = forward_pass(model, inputs, loss_fn, reduce_to_vector)
    gramian = engine.compute_gramian(losses)

    assert_close(gramian, autograd_gramian)


@mark.parametrize(["factory", "batch_size"], PARAMETRIZATIONS)
def test_iwrm_steps_with_autogram(factory: ModuleFactory, batch_size: int):
    """Tests that the autogram engine doesn't raise any error during several IWRM iterations."""

    n_iter = 3
    model = factory()
    weighting = UPGradWeighting()
    engine = Engine(model)
    optimizer = SGD(model.parameters(), lr=1e-7)

    for i in range(n_iter):
        inputs, targets = make_inputs_and_targets(model, batch_size)
        loss_fn = make_mse_loss_fn(targets)
        autogram_forward_backward(model, inputs, loss_fn, engine, weighting)
        optimizer.step()
        model.zero_grad()


@mark.parametrize(["factory", "batch_size"], BASE_PARAMETRIZATIONS)
@mark.parametrize("use_engine", [False, True])
def test_autograd_while_modules_are_hooked(
    factory: ModuleFactory,
    batch_size: int,
    use_engine: bool,
):
    """
    Tests that the hooks added when constructing the engine do not interfere with a simple autograd
    call.
    """

    model, model_autogram = factory(), factory()
    inputs, targets = make_inputs_and_targets(model, batch_size)
    loss_fn = make_mse_loss_fn(targets)

    autograd_forward_backward(model, inputs, loss_fn)
    autograd_grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}

    # Hook modules and optionally compute the Gramian
    engine = Engine(model_autogram)

    if use_engine:
        losses = forward_pass(model_autogram, inputs, loss_fn, reduce_to_vector)
        _ = engine.compute_gramian(losses)
    # Verify that even with the hooked modules, autograd works normally when not using the engine.
    # Results should be the same as a normal call to autograd, and no time should be spent computing
    # the gramian at all.
    autograd_forward_backward(model_autogram, inputs, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}

    assert_tensor_dicts_are_close(grads, autograd_grads)
    assert engine._gramian_accumulator.gramian is None


def test_compute_gramian_manual():
    """
    Tests that the Gramian computed by the `Engine` equals to a manual computation of the expected
    Gramian.
    """

    in_dims = 18
    out_dims = 25
    factory = ModuleFactory(Linear, in_dims, out_dims)
    model = factory()
    input = randn_(in_dims)

    engine = Engine(model)
    output = model(input)
    gramian = engine.compute_gramian(output)

    # Compute the expected gramian
    weight_jacobian = zeros_([out_dims, model.weight.numel()])
    for j in range(out_dims):
        weight_jacobian[j, j * in_dims : (j + 1) * in_dims] = input
    weight_gramian = compute_gramian(weight_jacobian)
    bias_jacobian = torch.diag(ones_(out_dims))
    bias_gramian = compute_gramian(bias_jacobian)
    expected_gramian = weight_gramian + bias_gramian

    assert_close(gramian, expected_gramian)


@mark.parametrize(
    "shape",
    [
        [1, 2, 2, 3],
        [7, 3, 2, 5],
        [27, 6, 7],
        [3, 2, 1, 1],
        [3, 2, 1],
        [3, 2],
        [3],
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 1],
        [1],
    ],
)
def test_reshape_equivariance(shape: list[int]):
    """
    Test equivariance of `compute_gramian` under reshape operation. More precisely, if we reshape
    the `output` to some `shape`, then the result is the same as reshaping the Gramian to the
    corresponding shape.
    """

    input_size = shape[0]
    output_size = prod(shape[1:])
    factory = ModuleFactory(Linear, input_size, output_size)
    model1, model2 = factory(), factory()
    input = randn_([input_size])

    engine1 = Engine(model1)
    output = model1(input)
    gramian = engine1.compute_gramian(output)
    expected_reshaped_gramian = reshape_gramian(gramian, shape[1:])

    engine2 = Engine(model2)
    reshaped_output = model2(input).reshape(shape[1:])
    reshaped_gramian = engine2.compute_gramian(reshaped_output)

    assert_close(reshaped_gramian, expected_reshaped_gramian)


@mark.parametrize(
    ["shape", "source", "destination"],
    [
        ([50, 2, 2, 3], [0, 2], [1, 0]),
        ([60, 3, 2, 5], [1], [2]),
        ([30, 6, 7], [0, 1], [1, 0]),
        ([3, 2], [0], [0]),
        ([3], [], []),
        ([3, 2, 1], [1, 0], [0, 1]),
        ([4, 3, 2], [], []),
        ([1, 1, 1], [1, 0], [0, 1]),
    ],
)
def test_movedim_equivariance(shape: list[int], source: list[int], destination: list[int]):
    """
    Test equivariance of `compute_gramian` under movedim operation. More precisely, if we movedim
    the `output` on some dimensions, then the result is the same as movedim on the Gramian with the
    corresponding dimensions.
    """

    input_size = shape[0]
    output_size = prod(shape[1:])
    factory = ModuleFactory(Linear, input_size, output_size)
    model1, model2 = factory(), factory()
    input = randn_([input_size])

    engine1 = Engine(model1)
    output = model1(input).reshape(shape[1:])
    gramian = engine1.compute_gramian(output)
    expected_moved_gramian = movedim_gramian(gramian, source, destination)

    engine2 = Engine(model2)
    moved_output = model2(input).reshape(shape[1:]).movedim(source, destination)
    moved_gramian = engine2.compute_gramian(moved_output)

    assert_close(moved_gramian, expected_moved_gramian)
