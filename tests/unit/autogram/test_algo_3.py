import time

import torch
from pytest import mark
from torch import Tensor, nn
from torch.nn import ReLU
from unit._utils import randint_, randn_
from unit.autojac._transform._dict_assertions import assert_tensor_dicts_are_close
from unit.conftest import DEVICE

from torchjd import backward
from torchjd._autogram._rev_gram_acc import autogram_forward_backward
from torchjd.aggregation import Aggregator, Mean, UPGrad


class SmartFlatten(nn.Module):
    """
    Flatten reducing inputs of shape [N, H, W, C] into [N, H * W * C] or reducing inputs of shape
    [H, W, C] into [H * W * C].
    """

    def forward(self, input):
        if input.dim() == 4:
            return torch.flatten(input, start_dim=1)
        elif input.dim() == 3:
            return torch.flatten(input)
        else:
            raise ValueError(f"Unsupported number of dimensions: {input.dim()}")


class Cifar10Model(nn.Sequential):
    def __init__(self):
        layers = [
            nn.Conv2d(3, 32, 3),
            ReLU(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.Sequential(nn.MaxPool2d(2), ReLU()),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.Sequential(nn.MaxPool2d(3), ReLU(), SmartFlatten()),
            nn.Linear(1024, 128),
            ReLU(),
            nn.Linear(128, 10),
        ]
        super().__init__(*layers)


class FlatNonSequentialNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(9, 1024)
        self.fc1 = nn.Linear(1024, 1025)
        self.fc2 = nn.Linear(1025, 1026)
        self.fc3 = nn.Linear(1024, 1026)

    def forward(self, input: Tensor) -> Tensor:
        common_input = self.relu(self.fc0(input))
        branch1 = self.fc2(self.relu(self.fc1(common_input)))
        branch2 = self.fc3(common_input)
        output = branch1 + branch2
        return output


@mark.parametrize(
    ["model", "single_input_shape"], [(Cifar10Model(), (3, 32, 32)), (FlatNonSequentialNN(), (9,))]
)
def test_speed(model: nn.Module, single_input_shape: tuple[int, ...]):
    batch_size = 64
    input_shape = (batch_size,) + single_input_shape
    input = randn_(input_shape)
    target = randint_(0, 10, (batch_size,))

    model = model.to(device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    A = Mean()
    W = A.weighting

    print(f"\nTimes for forward + backward with BS={batch_size}, A={A} on {DEVICE}.")

    torch.cuda.empty_cache()
    autograd_forward_backward(model, criterion, input, target)

    for i in range(10):
        model.zero_grad()
        with Timer("autograd"):
            autograd_forward_backward(model, criterion, input, target)

    torch.cuda.empty_cache()
    autojac_forward_backward(model, criterion, input, target, A)

    for i in range(10):
        model.zero_grad()
        with Timer("autojac"):
            autojac_forward_backward(model, criterion, input, target, A)

    torch.cuda.empty_cache()
    autogram_forward_backward(model, criterion, input, target, W)

    for i in range(10):
        model.zero_grad()
        with Timer("autogram"):
            autogram_forward_backward(model, criterion, input, target, W)


@mark.parametrize(
    ["model", "single_input_shape"], [(Cifar10Model(), (3, 32, 32)), (FlatNonSequentialNN(), (9,))]
)
def test_equivalence(model: nn.Module, single_input_shape: tuple[int, ...]):
    batch_size = 64
    input_shape = (batch_size,) + single_input_shape
    input = randn_(input_shape)
    target = randint_(0, 10, (batch_size,))

    model = model.to(device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    A = UPGrad()
    W = A.weighting.weighting

    autojac_forward_backward(model, criterion, input, target, A)
    expected_grads = {p: p.grad for p in model.parameters() if p.grad is not None}
    model.zero_grad()

    autogram_forward_backward(model, criterion, input, target, W)
    grads = {p: p.grad for p in model.parameters() if p.grad is not None}

    assert_tensor_dicts_are_close(grads, expected_grads)


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
    model: nn.Module,
    criterion: nn.Module,
    input: Tensor,
    target: Tensor,
    aggregator: Aggregator,
) -> None:
    output = model(input)
    losses = criterion(output, target)
    backward(losses, aggregator=aggregator)


def autograd_forward_backward(
    model: nn.Module,
    criterion: nn.Module,
    input: Tensor,
    target: Tensor,
) -> None:
    output = model(input)
    losses = criterion(output, target)
    loss = losses.mean()
    loss.backward()
