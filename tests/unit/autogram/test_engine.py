from itertools import combinations

import pytest
import torch
from pytest import mark, param
from torch import nn
from torch.optim import SGD
from unit.conftest import DEVICE
from utils.architectures import (
    AlexNet,
    Cifar10Model,
    FreeParam,
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
    WithBuffered,
    WithModuleTrackingRunningStats,
    WithNoTensorOutput,
    WithRNN,
    WithSideEffect,
    WithSomeFrozenModule,
)
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.forward_backwards import (
    autograd_forward_backward,
    autogram_forward_backward,
    autojac_forward_backward,
    make_mse_loss_fn,
)
from utils.tensors import make_tensors

from torchjd.aggregation import (
    IMTLG,
    MGDA,
    Aggregator,
    AlignedMTL,
    AlignedMTLWeighting,
    DualProj,
    DualProjWeighting,
    IMTLGWeighting,
    Mean,
    MeanWeighting,
    MGDAWeighting,
    PCGrad,
    PCGradWeighting,
    Random,
    RandomWeighting,
    Sum,
    SumWeighting,
    UPGrad,
    UPGradWeighting,
    Weighting,
)
from torchjd.autogram._engine import Engine
from torchjd.autojac._transform import Diagonalize, Init, Jac, OrderedSet
from torchjd.autojac._transform._aggregate import _Matrixify

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
    param(Cifar10Model, 32, marks=[mark.slow, mark.garbage_collect]),
    param(AlexNet, 4, marks=[mark.slow, mark.garbage_collect]),
    param(InstanceNormResNet18, 8, marks=[mark.slow, mark.garbage_collect]),
]

AGGREGATORS_AND_WEIGHTINGS: list[tuple[Aggregator, Weighting]] = [
    (UPGrad(), UPGradWeighting()),
    (AlignedMTL(), AlignedMTLWeighting()),
    (DualProj(), DualProjWeighting()),
    (IMTLG(), IMTLGWeighting()),
    (Mean(), MeanWeighting()),
    (MGDA(), MGDAWeighting()),
    (PCGrad(), PCGradWeighting()),
    (Random(), RandomWeighting()),
    (Sum(), SumWeighting()),
]

try:
    from torchjd.aggregation import CAGrad, CAGradWeighting

    AGGREGATORS_AND_WEIGHTINGS.append((CAGrad(c=0.5), CAGradWeighting(c=0.5)))
except ImportError:
    pass

WEIGHTINGS = [pair[1] for pair in AGGREGATORS_AND_WEIGHTINGS]


@mark.parametrize(["architecture", "batch_size"], PARAMETRIZATIONS)
@mark.parametrize(["aggregator", "weighting"], AGGREGATORS_AND_WEIGHTINGS)
def test_equivalence(
    architecture: type[ShapedModule],
    batch_size: int,
    aggregator: Aggregator,
    weighting: Weighting,
):
    n_iter = 3

    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    torch.manual_seed(0)
    model_autojac = architecture().to(device=DEVICE)
    torch.manual_seed(0)
    model_autogram = architecture().to(device=DEVICE)

    engine = Engine(model_autogram.modules())
    optimizer_autojac = SGD(model_autojac.parameters(), lr=1e-7)
    optimizer_autogram = SGD(model_autogram.parameters(), lr=1e-7)

    for i in range(n_iter):
        inputs = make_tensors(batch_size, input_shapes)
        targets = make_tensors(batch_size, output_shapes)
        loss_fn = make_mse_loss_fn(targets)

        torch.random.manual_seed(0)  # Fix randomness for random aggregators and random models
        autojac_forward_backward(model_autojac, inputs, loss_fn, aggregator)
        expected_grads = {
            name: p.grad for name, p in model_autojac.named_parameters() if p.grad is not None
        }

        torch.random.manual_seed(0)  # Fix randomness for random weightings and random models
        autogram_forward_backward(model_autogram, engine, weighting, inputs, loss_fn)
        grads = {
            name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None
        }

        assert_tensor_dicts_are_close(grads, expected_grads)

        optimizer_autojac.step()
        model_autojac.zero_grad()

        optimizer_autogram.step()
        model_autogram.zero_grad()


@mark.parametrize(["architecture", "batch_size"], PARAMETRIZATIONS)
def test_autograd_while_modules_are_hooked(architecture: type[ShapedModule], batch_size: int):
    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    W = UPGradWeighting()
    A = UPGrad()
    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.manual_seed(0)
    model = architecture().to(device=DEVICE)

    torch.manual_seed(0)  # Fix randomness for random models
    autojac_forward_backward(model, input, loss_fn, A)
    autojac_grads = {
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

    # Hook modules and verify that we're equivalent to autojac when using the engine
    engine = Engine(model_autogram.modules())
    torch.manual_seed(0)  # Fix randomness for random models
    autogram_forward_backward(model_autogram, engine, W, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autojac_grads)
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


@mark.parametrize("weighting", WEIGHTINGS)
@mark.parametrize("gramian_module_names", _non_empty_subsets({"fc0", "fc1", "fc2", "fc3", "fc4"}))
def test_partial_autogram(weighting: Weighting, gramian_module_names: set[str]):
    """
    Tests that partial JD via the autogram engine works similarly as if the gramian was computed via
    the autojac engine.

    Note that this test is a bit redundant now that we have the Engine interface, because it now
    just compares two ways of computing the Gramian, which is independant of the idea of partial JD.
    """

    architecture = SimpleBranched
    batch_size = 64

    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.manual_seed(0)
    model = architecture().to(device=DEVICE)

    output = model(input)
    losses = loss_fn(output)
    losses_ = OrderedSet(losses)

    init = Init(losses_)
    diag = Diagonalize(losses_)

    gramian_modules = [model.get_submodule(name) for name in gramian_module_names]
    gramian_params = OrderedSet({})
    for m in gramian_modules:
        gramian_params += OrderedSet(m.parameters())

    jac = Jac(losses_, OrderedSet(gramian_params), None, True)
    mat = _Matrixify()
    transform = mat << jac << diag << init

    jacobian_matrices = transform({})
    jacobian_matrix = torch.cat(list(jacobian_matrices.values()), dim=1)
    gramian = jacobian_matrix @ jacobian_matrix.T
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
    model = architecture().to(device=DEVICE)

    with pytest.raises(ValueError):
        _ = Engine(model.modules())


def test_non_vector_input_to_compute_gramian():
    architecture = Cifar10Model
    batch_size = 64

    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.manual_seed(0)
    model = architecture().to(device=DEVICE)

    engine = Engine(model.modules())

    output = model(input)
    losses = loss_fn(output).reshape([8, 8])

    with pytest.raises(ValueError):
        engine.compute_gramian(losses)


def test_non_batched():
    # This is an adaptation of basic example using autogram.
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
    optimizer = SGD(model.parameters(), lr=0.1)

    engine = Engine(model.modules(), False)

    weighting = UPGradWeighting()
    input = torch.randn(16, 10)  # Batch of 16 random input vectors of length 10
    target1 = torch.randn(16)  # First batch of 16 targets
    target2 = torch.randn(16)  # Second batch of 16 targets

    loss_fn = MSELoss()
    output = model(input)
    loss1 = loss_fn(output[:, 0], target1)
    loss2 = loss_fn(output[:, 1], target2)
    losses = torch.stack([loss1, loss2])

    optimizer.zero_grad()
    gramian = engine.compute_gramian(losses)
    losses.backward(weighting(gramian))
    optimizer.step()
