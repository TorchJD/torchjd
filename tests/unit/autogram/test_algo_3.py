import time
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import Parameter, ReLU
from torch.testing import assert_close
from torch.utils._ordered_set import OrderedSet

from torchjd._autojac._transform import Diagonalize, Init, Jac
from torchjd._autojac._transform._aggregate import _Matrixify

DEVICE = "cuda"
torch.set_default_device(DEVICE)


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
    torch.use_deterministic_algorithms(False)

    batch_size = 128
    input_shape = (batch_size, 3, 32, 32)
    input = torch.randn(input_shape)
    target = torch.randint(0, 10, (batch_size,))

    model = Cifar10Model()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # Compute gramian to initialize everything correctly, prior to making the timed call
    activations = compute_activations(criterion, input, model, target)
    _ = autogram_(activations, batch_size, criterion, model)

    # Re-compute the activations because autogram_ needs the non-freed graph
    activations = compute_activations(criterion, input, model, target)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    gramian = autogram_(activations, batch_size, criterion, model)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    print(end - start)

    activations = compute_activations(criterion, input, model, target)
    _ = compute_gramian_via_autojac(activations, model)

    activations = compute_activations(criterion, input, model, target)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    expected_gramian = compute_gramian_via_autojac(activations, model)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(end - start)

    assert_close(gramian, expected_gramian)


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
    gramian = torch.zeros(batch_size, batch_size)
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


def test_diagonalize():
    t = torch.randn([2, 3])
    print(t)
    d = diagonalize(t)
    print(d.shape)
    print(d)


def diagonalize(t: Tensor) -> Tensor:
    d = torch.zeros((t.shape[0],) + t.shape)
    for i, row in enumerate(t):
        d[i, i] = row

    return d


def diagonalize_one_row(t: Tensor, i: int) -> Tensor:
    d = torch.zeros_like(t)
    d[i] = t[i]

    return d


def test_diagonalize_one_row_sparse():
    t = torch.randn(10, 5)
    print(diagonalize_one_row_sparse(t, 0))
    print(diagonalize_one_row_sparse(t, 1))
    print(diagonalize_one_row_sparse(t, 2))
    print(diagonalize_one_row_sparse(t, 3))


def diagonalize_one_row_sparse(x: Tensor, index: int = 0) -> Tensor:
    """
    Returns a sparse tensor with only the slice at `index` along dim=0 kept, in COO format.
    Works for x.ndim >= 1 of any shape.
    """
    dense = torch.zeros_like(x)
    dense[index] = x[index]
    sparse = dense.to_sparse()  # default is sparse_coo
    return sparse


def vjp_from_module(module: nn.Module, *inputs) -> Callable:
    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers())}
        return torch.func.functional_call(module, all_state, *inputs)

    return torch.func.vjp(functional_model_call, dict(module.named_parameters()))[1]
