import time

import torch
from torch import Tensor, nn
from torch.nn import ReLU
from torch.testing import assert_close
from unit._utils import randint_, randn_
from unit.conftest import DEVICE

from torchjd import backward
from torchjd._autogram._rev_gram_acc import autogram_forward_backward
from torchjd.aggregation import Aggregator, Mean


class Cifar10Model(nn.Sequential):
    def __init__(self):
        layers = [
            nn.Conv2d(3, 32, 3),
            ReLU(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.Sequential(nn.MaxPool2d(2), ReLU()),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.Sequential(nn.MaxPool2d(3), ReLU(), nn.Flatten()),
            nn.Linear(1024, 128),
            ReLU(),
            nn.Linear(128, 10),
        ]
        super().__init__(*layers)


def test_algo_3():
    batch_size = 64
    input_shape = (batch_size, 3, 32, 32)
    input = randn_(input_shape)
    target = randint_(0, 10, (batch_size,))

    model = Cifar10Model().to(device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    A = Mean()
    W = A.weighting

    print(f"\nTimes for forward + backward with BS={batch_size}, A={A} on {DEVICE}.")

    torch.cuda.empty_cache()
    autograd_forward_backward(model, criterion, input, target)
    model.zero_grad()
    with Timer("autograd"):
        autograd_forward_backward(model, criterion, input, target)

    torch.cuda.empty_cache()
    autojac_forward_backward(model, criterion, input, target, A)
    model.zero_grad()

    with Timer("autojac"):
        autojac_forward_backward(model, criterion, input, target, A)

    torch.cuda.empty_cache()
    autogram_forward_backward(model, criterion, input, target, W)
    model.zero_grad()

    with Timer("autogram"):
        autogram_forward_backward(model, criterion, input, target, W)


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.cuda_sync()
        self.start_time = time.perf_counter()
        return self  # This allows you to access the timer object within the 'with' block

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cuda_sync()
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name}: {self.elapsed_time:.4f} seconds")

    @staticmethod
    def cuda_sync():
        if str(DEVICE).startswith("cuda"):
            torch.cuda.synchronize()


def autojac_forward_backward(
    model: nn.Sequential,
    criterion: nn.Module,
    input: Tensor,
    target: Tensor,
    aggregator: Aggregator,
) -> None:
    output = model(input)
    losses = criterion(output, target)
    backward(losses, aggregator=aggregator)


def autograd_forward_backward(
    model: nn.Sequential,
    criterion: nn.Module,
    input: Tensor,
    target: Tensor,
) -> None:
    output = model(input)
    losses = criterion(output, target)
    loss = losses.mean()
    loss.backward()


def test_sequential():
    hidden_size = 1000
    n_layers = 30
    layers = [
        nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(n_layers)
    ]
    # layers = [nn.Linear(3, 4), nn.Linear(4, 5)]

    model = nn.Sequential(*layers).to(DEVICE)

    input = randn_((16, hidden_size))

    output = model(input).mean()
    torch.autograd.grad(output, list(model.parameters()))

    output = model(input).mean()
    with Timer("autograd all together"):
        expected_result = torch.autograd.grad(output, list(model.parameters()))

    outputs = _compute_outputs(input, model)
    autograd_chained_manually(layers, outputs)

    outputs = _compute_outputs(input, model)
    with Timer("autograd chained manually"):
        result = autograd_chained_manually(layers, outputs)

    assert len(result) == len(expected_result)
    for grad, expected_grad in zip(result, expected_result):
        assert_close(grad, expected_grad)


def autograd_chained_manually(layers, outputs):
    result = []
    grad_output = torch.ones_like(outputs[-1]) / outputs[-1].numel()
    for i, (input, output, layer) in list(enumerate(zip(outputs[:-1], outputs[1:], layers)))[::-1]:
        grad_wrt_param = torch.autograd.grad(
            output, list(layer.parameters()), grad_outputs=grad_output, retain_graph=True
        )
        result.extend(reversed([*grad_wrt_param]))

        if i == 0:
            break
        grad_output = torch.autograd.grad(
            output, input, grad_outputs=grad_output, retain_graph=False
        )[0]

    return tuple(reversed(result))


def _compute_outputs(input, model: nn.Sequential) -> list[Tensor]:
    activations = [input]
    for layer in model:
        activation = layer(activations[-1])
        activations.append(activation)

    return activations
