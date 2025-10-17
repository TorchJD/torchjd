from collections.abc import Callable
from itertools import combinations
from math import prod

import pytest
import torch
from pytest import mark, param
from torch import Tensor
from torch.nn import RNN, BatchNorm2d, InstanceNorm2d, Linear
from torch.optim import SGD
from torch.testing import assert_close
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
    SimpleParamReuse,
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
    WithSideEffect,
    WithSomeFrozenModule,
    WithTransformer,
    WithTransformerLarge,
    get_in_out_shapes,
)
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.forward_backwards import (
    autograd_forward_backward,
    autogram_forward_backward,
    compute_gramian,
    compute_gramian_with_autograd,
    make_mse_loss_fn,
    reduce_to_first_tensor,
    reduce_to_matrix,
    reduce_to_scalar,
    reduce_to_vector,
)
from utils.tensors import make_tensors, ones_, randn_, zeros_

from torchjd.aggregation import UPGradWeighting
from torchjd.autogram._engine import Engine
from torchjd.autogram._gramian_utils import movedim_gramian, reshape_gramian

PARAMETRIZATIONS = [
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
    (ModuleFactory(SimpleParamReuse), 32),
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


def _assert_gramian_is_equivalent_to_autograd(
    factory: ModuleFactory, batch_size: int, batch_dim: int | None
):
    model_autograd, model_autogram = factory(), factory()
    input_shapes, output_shapes = get_in_out_shapes(model_autograd)

    engine = Engine(model_autogram, batch_dim=batch_dim)

    inputs = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.random.manual_seed(0)  # Fix randomness for random models
    output = model_autograd(inputs)
    losses = reduce_to_vector(loss_fn(output))
    autograd_gramian = compute_gramian_with_autograd(losses, list(model_autograd.parameters()))

    torch.random.manual_seed(0)  # Fix randomness for random models
    output = model_autogram(inputs)
    losses = reduce_to_vector(loss_fn(output))
    autogram_gramian = engine.compute_gramian(losses)

    assert_close(autogram_gramian, autograd_gramian, rtol=1e-4, atol=3e-5)


@mark.parametrize(["factory", "batch_size"], PARAMETRIZATIONS)
@mark.parametrize("batch_dim", [0, None])
def test_compute_gramian(factory: ModuleFactory, batch_size: int, batch_dim: int | None):
    """Tests that the autograd and the autogram engines compute the same gramian."""

    _assert_gramian_is_equivalent_to_autograd(factory, batch_size, batch_dim)


@mark.parametrize(
    "factory",
    [
        ModuleFactory(BatchNorm2d, num_features=3, affine=True, track_running_stats=False),
        ModuleFactory(WithSideEffect),
        ModuleFactory(Randomness),
        ModuleFactory(InstanceNorm2d, num_features=3, affine=True, track_running_stats=True),
        param(
            ModuleFactory(RNN, input_size=8, hidden_size=5, batch_first=True),
            marks=mark.xfail_if_cuda,
        ),
    ],
)
@mark.parametrize("batch_size", [1, 3, 32])
@mark.parametrize("batch_dim", [param(0, marks=mark.xfail), None])
def test_compute_gramian_with_weird_modules(
    factory: ModuleFactory, batch_size: int, batch_dim: int | None
):
    """
    Tests that compute_gramian works even with some problematic modules when batch_dim is None. It
    is expected to fail on those when the engine uses the batched optimization (when batch_dim=0).
    """

    _assert_gramian_is_equivalent_to_autograd(factory, batch_size, batch_dim)


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
@mark.parametrize("batch_dim", [0, None])
def test_compute_gramian_unsupported_architectures(
    factory: ModuleFactory, batch_size: int, batch_dim: int | None
):
    """
    Tests compute_gramian on some architectures that are known to be unsupported. It is expected to
    fail.
    """

    _assert_gramian_is_equivalent_to_autograd(factory, batch_size, batch_dim)


@mark.parametrize("batch_size", [1, 3, 16])
@mark.parametrize(
    ["reduction", "movedim_source", "movedim_destination", "batch_dim"],
    [
        # 0D
        (reduce_to_scalar, [], [], None),  # ()
        # 1D
        (reduce_to_vector, [], [], 0),  # (batch_size,)
        (reduce_to_vector, [], [], None),  # (batch_size,)
        # 2D
        (reduce_to_matrix, [], [], 0),  # (batch_size, d1 * d2)
        (reduce_to_matrix, [], [], None),  # (batch_size, d1 * d2)
        (reduce_to_matrix, [0], [1], 1),  # (d1 * d2, batch_size)
        (reduce_to_matrix, [0], [1], None),  # (d1 * d2, batch_size)
        # 3D
        (reduce_to_first_tensor, [], [], 0),  # (batch_size, d1, d2)
        (reduce_to_first_tensor, [], [], None),  # (batch_size, d1, d2)
        (reduce_to_first_tensor, [0], [1], 1),  # (d1, batch_size, d2)
        (reduce_to_first_tensor, [0], [1], None),  # (d1, batch_size, d2)
        (reduce_to_first_tensor, [0], [2], 2),  # (d2, d1, batch_size)
        (reduce_to_first_tensor, [0], [2], None),  # (d2, d1, batch_size)
    ],
)
def test_compute_gramian_various_output_shapes(
    batch_size: int | None,
    reduction: Callable[[list[Tensor]], Tensor],
    batch_dim: int | None,
    movedim_source: list[int],
    movedim_destination: list[int],
):
    """
    Tests that the autograd and the autogram engines compute the same gramian when the output can
    have various different shapes, and can be batched in any of its dimensions.
    """

    factory = ModuleFactory(Ndim2Output)
    model_autograd, model_autogram = factory(), factory()
    input_shapes, output_shapes = get_in_out_shapes(model_autograd)

    engine = Engine(model_autogram, batch_dim=batch_dim)

    inputs = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.random.manual_seed(0)  # Fix randomness for random models
    output = model_autograd(inputs)
    losses = reduction(loss_fn(output))
    reshaped_losses = torch.movedim(losses, movedim_source, movedim_destination)
    # Go back to a vector so that compute_gramian_with_autograd works
    loss_vector = reshaped_losses.reshape([-1])
    autograd_gramian = compute_gramian_with_autograd(loss_vector, list(model_autograd.parameters()))
    expected_gramian = reshape_gramian(autograd_gramian, list(reshaped_losses.shape))

    torch.random.manual_seed(0)  # Fix randomness for random models
    output = model_autogram(inputs)
    losses = reduction(loss_fn(output))
    reshaped_losses = torch.movedim(losses, movedim_source, movedim_destination)
    autogram_gramian = engine.compute_gramian(reshaped_losses)

    assert_close(autogram_gramian, expected_gramian, rtol=1e-4, atol=1e-5)


def _non_empty_subsets(elements: set) -> list[set]:
    """
    Generates the list of subsets of the given set, excluding the empty set.
    """
    return [set(c) for r in range(1, len(elements) + 1) for c in combinations(elements, r)]


@mark.parametrize("gramian_module_names", _non_empty_subsets({"fc0", "fc1", "fc2", "fc3", "fc4"}))
@mark.parametrize("batch_dim", [0, None])
def test_compute_partial_gramian(gramian_module_names: set[str], batch_dim: int | None):
    """
    Tests that the autograd and the autogram engines compute the same gramian when only a subset of
    the model parameters is specified.
    """

    factory = ModuleFactory(SimpleBranched)
    model = factory()
    input_shapes, output_shapes = get_in_out_shapes(model)
    batch_size = 64

    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    output = model(input)
    losses = reduce_to_vector(loss_fn(output))

    gramian_modules = [model.get_submodule(name) for name in gramian_module_names]
    gramian_params = []
    for m in gramian_modules:
        gramian_params += list(m.parameters())

    autograd_gramian = compute_gramian_with_autograd(losses, gramian_params, retain_graph=True)
    torch.manual_seed(0)

    engine = Engine(*gramian_modules, batch_dim=batch_dim)

    output = model(input)
    losses = reduce_to_vector(loss_fn(output))
    gramian = engine.compute_gramian(losses)

    assert_close(gramian, autograd_gramian)


@mark.parametrize(["factory", "batch_size"], PARAMETRIZATIONS)
@mark.parametrize("batch_dim", [0, None])
def test_iwrm_steps_with_autogram(factory: ModuleFactory, batch_size: int, batch_dim: int | None):
    """Tests that the autogram engine doesn't raise any error during several IWRM iterations."""

    n_iter = 3

    model = factory()
    input_shapes, output_shapes = get_in_out_shapes(model)

    weighting = UPGradWeighting()

    engine = Engine(model, batch_dim=batch_dim)
    optimizer = SGD(model.parameters(), lr=1e-7)

    for i in range(n_iter):
        inputs = make_tensors(batch_size, input_shapes)
        targets = make_tensors(batch_size, output_shapes)
        loss_fn = make_mse_loss_fn(targets)

        autogram_forward_backward(model, engine, weighting, inputs, loss_fn)

        optimizer.step()
        model.zero_grad()


@mark.parametrize(["factory", "batch_size"], PARAMETRIZATIONS)
@mark.parametrize("use_engine", [False, True])
@mark.parametrize("batch_dim", [0, None])
def test_autograd_while_modules_are_hooked(
    factory: ModuleFactory, batch_size: int, use_engine: bool, batch_dim: int | None
):
    """
    Tests that the hooks added when constructing the engine do not interfere with a simple autograd
    call.
    """

    model, model_autogram = factory(), factory()
    input_shapes, output_shapes = get_in_out_shapes(model)

    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.manual_seed(0)  # Fix randomness for random models
    autograd_forward_backward(model, input, loss_fn)
    autograd_grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}

    # Hook modules and optionally compute the Gramian
    engine = Engine(model_autogram, batch_dim=batch_dim)
    if use_engine:
        torch.manual_seed(0)  # Fix randomness for random models
        output = model_autogram(input)
        losses = reduce_to_vector(loss_fn(output))
        _ = engine.compute_gramian(losses)

    # Verify that even with the hooked modules, autograd works normally when not using the engine.
    # Results should be the same as a normal call to autograd, and no time should be spent computing
    # the gramian at all.
    torch.manual_seed(0)  # Fix randomness for random models
    autograd_forward_backward(model_autogram, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}

    assert_tensor_dicts_are_close(grads, autograd_grads)
    assert engine._gramian_accumulator.gramian is None


@mark.parametrize(
    ["factory", "batch_dim"],
    [
        (ModuleFactory(InstanceNorm2d, num_features=3, affine=True, track_running_stats=True), 0),
        (ModuleFactory(RNN, input_size=8, hidden_size=5, batch_first=True), 0),
        (ModuleFactory(BatchNorm2d, num_features=3, affine=True, track_running_stats=False), 0),
    ],
)
def test_incompatible_modules(factory: ModuleFactory, batch_dim: int | None):
    """Tests that the engine cannot be constructed with incompatible modules."""

    model = factory()
    with pytest.raises(ValueError):
        _ = Engine(model, batch_dim=batch_dim)


def test_compute_gramian_manual():
    """
    Tests that the Gramian computed by the `Engine` equals to a manual computation of the expected
    Gramian.
    """

    in_dims = 18
    out_dims = 25

    factory = ModuleFactory(Linear, in_dims, out_dims)
    model = factory()
    engine = Engine(model, batch_dim=None)

    input = randn_(in_dims)
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

    engine1 = Engine(model1, batch_dim=None)
    engine2 = Engine(model2, batch_dim=None)

    input = randn_([input_size])
    output = model1(input)
    reshaped_output = model2(input).reshape(shape[1:])

    gramian = engine1.compute_gramian(output)
    reshaped_gramian = engine2.compute_gramian(reshaped_output)
    expected_reshaped_gramian = reshape_gramian(gramian, shape[1:])

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

    engine1 = Engine(model1, batch_dim=None)
    engine2 = Engine(model2, batch_dim=None)

    input = randn_([input_size])
    output = model1(input).reshape(shape[1:])
    moved_output = model2(input).reshape(shape[1:]).movedim(source, destination)

    gramian = engine1.compute_gramian(output)
    moved_gramian = engine2.compute_gramian(moved_output)
    expected_moved_gramian = movedim_gramian(gramian, source, destination)

    assert_close(moved_gramian, expected_moved_gramian)


@mark.parametrize(
    ["shape", "batch_dim"],
    [
        ([2, 5, 3, 2], 2),
        ([3, 2, 5], 1),
        ([6, 3], 0),
        ([4, 3, 2], 1),
        ([1, 1, 1], 0),
        ([1, 1, 1], 1),
        ([1, 1, 1], 2),
        ([1, 1], 0),
        ([1], 0),
        ([4, 3, 1], 2),
    ],
)
def test_batched_non_batched_equivalence(shape: list[int], batch_dim: int):
    """
    Tests that for a vector with some batched dimensions, the gramian is the same if we use the
    appropriate `batch_dim` or if we don't use any.
    """

    non_batched_shape = [shape[i] for i in range(len(shape)) if i != batch_dim]
    input_size = prod(non_batched_shape)
    batch_size = shape[batch_dim]
    output_size = input_size

    factory = ModuleFactory(Linear, input_size, output_size)
    model1, model2 = factory(), factory()

    engine1 = Engine(model1, batch_dim=batch_dim)
    engine2 = Engine(model2, batch_dim=None)

    input = randn_([batch_size, input_size])
    output1 = model1(input).reshape([batch_size] + non_batched_shape).movedim(0, batch_dim)
    output2 = model2(input).reshape([batch_size] + non_batched_shape).movedim(0, batch_dim)

    gramian1 = engine1.compute_gramian(output1)
    gramian2 = engine2.compute_gramian(output2)

    assert_close(gramian1, gramian2)


@mark.parametrize(["factory", "batch_size"], PARAMETRIZATIONS)
def test_batched_non_batched_equivalence_2(factory: ModuleFactory, batch_size: int):
    """
    Same as test_batched_non_batched_equivalence but on real architectures, and thus only between
    batch_size=0 and batch_size=None.

    If for some architecture this test passes but the test_compute_gramian doesn't pass, it could be
    that the get_used_params does not work for some module of the architecture.
    """

    model_0, model_none = factory(), factory()
    input_shapes, output_shapes = get_in_out_shapes(model_0)

    engine_0 = Engine(model_0, batch_dim=0)
    engine_none = Engine(model_none, batch_dim=None)

    inputs = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.random.manual_seed(0)  # Fix randomness for random models
    output = model_0(inputs)
    losses_0 = reduce_to_vector(loss_fn(output))

    torch.random.manual_seed(0)  # Fix randomness for random models
    output = model_none(inputs)
    losses_none = reduce_to_vector(loss_fn(output))

    gramian_0 = engine_0.compute_gramian(losses_0)
    gramian_none = engine_none.compute_gramian(losses_none)

    assert_close(gramian_0, gramian_none, rtol=1e-4, atol=1e-5)
