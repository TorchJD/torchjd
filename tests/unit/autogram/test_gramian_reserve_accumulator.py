import pytest
import torch
from pytest import mark, param
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
    WithNoTensorOutput,
    WithSideEffect,
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
def test_augment_deaugment_reaugment(architecture: type[ShapedModule], batch_size: int):
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

    # Augment and verify that we're equivalent to autojac
    engine = Engine(model_autogram.modules())
    torch.manual_seed(0)  # Fix randomness for random models
    autogram_forward_backward(model_autogram, engine, W, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autojac_grads)
    model_autogram.zero_grad()

    # Verify that even with the hooked modules, autograd works normally
    torch.manual_seed(0)  # Fix randomness for random models
    autograd_forward_backward(model_autogram, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autograd_grads)
    model_autogram.zero_grad()

    # Re-augment and verify that we're still equivalent to autojac
    engine = Engine(model_autogram.modules())
    torch.manual_seed(0)  # Fix randomness for random models
    autogram_forward_backward(model_autogram, engine, W, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autojac_grads)


def test_partial_autogram():
    architecture1 = Cifar10Model.Body
    architecture2 = Cifar10Model.Head
    batch_size = 64

    input_shapes = architecture1.INPUT_SHAPES
    output_shapes = architecture2.OUTPUT_SHAPES

    W = UPGradWeighting()
    input = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    torch.manual_seed(0)
    model1 = architecture1().to(device=DEVICE)
    model2 = architecture2().to(device=DEVICE)

    output1 = model1(input)
    output2 = model2(output1)
    losses = loss_fn(output2)
    losses_ = OrderedSet(losses)

    init = Init(losses_)
    diag = Diagonalize(losses_)
    jac = Jac(losses_, OrderedSet(model2.parameters()), None, True)
    mat = _Matrixify()
    transform = mat << jac << diag << init

    jacobian_matrices = transform({})
    jacobian_matrix = torch.cat(list(jacobian_matrices.values()), dim=1)
    gramian = jacobian_matrix @ jacobian_matrix.T
    weights = W(gramian)

    loss = losses @ weights
    loss.backward()

    expected_grads1 = {name: p.grad for name, p in model1.named_parameters() if p.grad is not None}
    expected_grads2 = {name: p.grad for name, p in model2.named_parameters() if p.grad is not None}
    model1.zero_grad()
    model2.zero_grad()

    engine = Engine(model2.modules())

    output = model1(input)
    output = model2(output)
    losses = loss_fn(output)
    gramian = engine.compute_gramian(losses)
    losses.backward(W(gramian))

    grads1 = {name: p.grad for name, p in model1.named_parameters() if p.grad is not None}
    grads2 = {name: p.grad for name, p in model2.named_parameters() if p.grad is not None}

    assert_tensor_dicts_are_close(grads1, expected_grads1)
    assert_tensor_dicts_are_close(grads2, expected_grads2)


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


def test_autograd_backward_on_augmented_model():
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

    torch.autograd.backward(losses, torch.ones_like(losses))

    # A call to autograd.backward phase should not compute the gramian.
    assert engine._gramian_accumulator.gramian is None
