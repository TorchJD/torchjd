import torch
from pytest import mark, param
from torch.optim import SGD
from unit.conftest import DEVICE
from utils.architectures import (
    Cifar10Model,
    FreeParam,
    InstanceNormResNet18,
    InterModuleParamReuse,
    MIMOBranched,
    MISOBranched,
    ModuleReuse,
    MultiInputMultiOutput,
    MultiInputSingleOutput,
    NoFreeParam,
    OverlyNested,
    PIPOBranched,
    PISOBranched,
    PyTreeInputPyTreeOutput,
    PyTreeInputSingleOutput,
    ShapedModule,
    SimpleBranched,
    SimpleParamReuse,
    SingleInputPyTreeOutput,
    SIPOBranched,
    SomeFrozenParam,
    SomeUnusedParam,
    WithBuffered,
    WithNoTensorOutput,
)
from utils.dict_assertions import assert_tensor_dicts_are_close
from utils.forward_backwards import (
    autograd_forward_backward,
    autogram_forward_backward,
    autojac_forward_backward,
    make_mse_loss_fn,
)
from utils.tensors import make_tensors

from torchjd._autogram._augment_model import augment_model_for_gramian_based_iwrm
from torchjd._autojac._transform import Diagonalize, Init, Jac, OrderedSet
from torchjd._autojac._transform._aggregate import _Matrixify
from torchjd.aggregation import UPGrad, UPGradWrapper

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
    (FreeParam, 32),
    (NoFreeParam, 32),
    param(Cifar10Model, 32, marks=mark.slow),
    param(InstanceNormResNet18, 8, marks=mark.slow),
]


@mark.parametrize(["architecture", "batch_size"], PARAMETRIZATIONS)
def test_equivalence(architecture: type[ShapedModule], batch_size: int):
    n_iter = 3

    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    W = UPGradWrapper()
    A = UPGrad()

    torch.manual_seed(0)
    model_autojac = architecture().to(device=DEVICE)
    torch.manual_seed(0)
    model_autogram = architecture().to(device=DEVICE)

    augment_model_for_gramian_based_iwrm(model_autogram, W)
    optimizer_autojac = SGD(model_autojac.parameters(), lr=1e-7)
    optimizer_autogram = SGD(model_autogram.parameters(), lr=1e-7)

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


@mark.parametrize(["architecture", "batch_size"], PARAMETRIZATIONS)
def test_augment_deaugment_reaugment(architecture: type[ShapedModule], batch_size: int):
    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES

    W = UPGradWrapper()
    A = UPGrad()
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
    handle = augment_model_for_gramian_based_iwrm(model_autogram, W)
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
    augment_model_for_gramian_based_iwrm(model_autogram, W)
    autogram_forward_backward(model_autogram, input, loss_fn)
    grads = {name: p.grad for name, p in model_autogram.named_parameters() if p.grad is not None}
    assert_tensor_dicts_are_close(grads, autojac_grads)


def test_partial_autogram():
    architecture1 = Cifar10Model.Body
    architecture2 = Cifar10Model.Head
    batch_size = 64

    input_shapes = architecture1.INPUT_SHAPES
    output_shapes = architecture2.OUTPUT_SHAPES

    W = UPGradWrapper()
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

    augment_model_for_gramian_based_iwrm(model2, W)

    output = model1(input)
    output = model2(output)
    losses = loss_fn(output)
    losses.backward(torch.ones_like(losses))

    grads1 = {name: p.grad for name, p in model1.named_parameters() if p.grad is not None}
    grads2 = {name: p.grad for name, p in model2.named_parameters() if p.grad is not None}

    assert_tensor_dicts_are_close(grads1, expected_grads1)
    assert_tensor_dicts_are_close(grads2, expected_grads2)
