import time
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import Parameter, ReLU
from torch.testing import assert_close
from torch.utils._ordered_set import OrderedSet
from unit.conftest import DEVICE

from torchjd._autojac._transform import Diagonalize, Init, Jac
from torchjd._autojac._transform._aggregate import _Matrixify


class Cifar10Model(nn.Sequential):
    def __init__(self):
        layers = [
            nn.Conv2d(3, 32, 3),
            ReLU(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.MaxPool2d(2),
            ReLU(),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.MaxPool2d(3),
            ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            ReLU(),
            nn.Linear(128, 10),
        ]
        super().__init__(*layers)


def test_algo_3():
    batch_size = 128
    input_shape = (batch_size, 3, 32, 32)
    input = torch.randn(input_shape, device=DEVICE)
    target = torch.randint(0, 10, (batch_size,), device=DEVICE)

    model = Cifar10Model().to(device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # Compute gramian to initialize everything correctly, prior to making the timed call
    # Re-compute the activations because autogram_ needs the non-freed graph
    activations = compute_activations(criterion, input, model, target)
    _ = autogram_(activations, batch_size, criterion, model)
    activations = compute_activations(criterion, input, model, target)
    with Timer(device=DEVICE):
        gramian = autogram_(activations, batch_size, criterion, model)

    activations = compute_activations(criterion, input, model, target)
    _ = compute_gramian_via_autojac(activations, model)
    activations = compute_activations(criterion, input, model, target)
    with Timer(device=DEVICE):
        expected_gramian = compute_gramian_via_autojac(activations, model)

    assert_close(gramian, expected_gramian)


class Timer:
    def __init__(self, device=None):
        self.device = device
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self  # This allows you to access the timer object within the 'with' block

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        print(f"Elapsed time: {self.elapsed_time:.4f} seconds")


def compute_gramian_via_autojac(activations, model: nn.Sequential):
    params = OrderedSet(model.parameters())
    output = OrderedSet([activations[-1]])
    init = Init(output)
    diag = Diagonalize(output)
    jac = Jac(output, params, chunk_size=1)
    matrixify = _Matrixify()
    transform = matrixify << jac << diag << init
    jacobian_matrices = transform({})
    expected_gramian = torch.stack([J @ J.T for J in jacobian_matrices.values()]).sum(dim=0)
    return expected_gramian


def compute_activations(criterion, input, model: nn.Sequential, target) -> list[Tensor]:
    activations = [input]
    for layer in model:
        activation = layer(activations[-1])
        activations.append(activation)

    losses = criterion(activations[-1], target)
    activations.append(losses)
    return activations


def autogram_(activations, batch_size, criterion, model: nn.Sequential):
    grad = torch.ones_like(activations[-1])
    gramian = torch.zeros(batch_size, batch_size, device=grad.device)
    for i, (input, output, layer) in list(
        enumerate(zip(activations[:-1], activations[1:], list(model) + [criterion]))
    )[::-1]:
        params = list(layer.parameters())
        if len(params) > 0:

            def get_vjp(input_j, grad_output_j) -> tuple[Tensor, ...]:
                return vjp_from_module(layer, input_j)(grad_output_j)

            jacobians = torch.vmap(get_vjp)(input, grad)

            assert len(jacobians) == 1

            for jacobian in jacobians[0].values():
                J = jacobian.reshape((batch_size, -1))
                gramian += J @ J.T  # Accumulate the gramian

        if i == 0:
            break  # Don't try to differentiate with respect to the model's input
        grad = torch.autograd.grad(output, input, grad, retain_graph=False)[0]

    return gramian


def vjp_from_module(module: nn.Module, *inputs) -> Callable:
    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers())}
        return torch.func.functional_call(module, all_state, *inputs)

    return torch.func.vjp(functional_model_call, dict(module.named_parameters()))[1]


def test_reused_tensor_vjp():
    param = torch.tensor(5.0)
    input = torch.tensor(2.0)

    def g(x, param_):
        return x * param_

    def f(x, param_):
        return x * param_**2

    def h(x, param_):
        return f(g(x, param_), param_)

    output, vjp = torch.func.vjp(h, input, param)
    grad_wrt_param_full = vjp(torch.ones_like(output))[1]
    assert_close(grad_wrt_param_full, input * 3 * param**2)

    output, vjp = torch.func.vjp(f, g(input, param), param)
    grad_wrt_param_partial = vjp(torch.ones_like(output))[1]
    assert_close(grad_wrt_param_partial, g(input, param) * 2 * param)


def test_reused_tensor_autograd():
    param = torch.tensor(5.0, requires_grad=True)
    input = torch.tensor(2.0)

    x1 = input * param
    x2 = x1 * param**2

    grad_wrt_param_full = torch.autograd.grad(x2, param)[0]
    assert_close(grad_wrt_param_full, input * 3 * param**2)

    x1 = input * param
    x2 = x1.detach() * param**2

    grad_wrt_param_partial = torch.autograd.grad(x2, param)[0]
    assert_close(grad_wrt_param_partial, x1 * 2 * param)
