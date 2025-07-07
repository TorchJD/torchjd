import time

import torch
from torch import Tensor, nn
from torch.nn import ReLU
from unit._utils import randint_, randn_
from unit.autojac._transform._dict_assertions import assert_tensor_dicts_are_close
from unit.conftest import DEVICE

from torchjd import backward
from torchjd._autogram._rev_gram_acc import autogram_forward_backward
from torchjd.aggregation import Aggregator, Mean, UPGrad


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


def test_speed():
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

    for i in range(20):
        model.zero_grad()
        autogram_forward_backward(model, criterion, input, target, W)
        model.zero_grad()

        with Timer("autogram"):
            autogram_forward_backward(model, criterion, input, target, W)


def test_equivalence():
    batch_size = 64
    input_shape = (batch_size, 3, 32, 32)
    input = randn_(input_shape)
    target = randint_(0, 10, (batch_size,))

    model = Cifar10Model().to(device=DEVICE)
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


def test_hook_autoremove():
    m = nn.Linear(3, 5)

    def hook(_, __):
        print("Calling hook")

    handle = m.register_full_backward_pre_hook(hook)

    def remove_hook(_, __, ___):
        handle.remove()

    m.register_full_backward_hook(remove_hook)

    for i in range(3):
        input = torch.randn((16, 3))
        output = m(input)
        output.backward(torch.ones_like(output))


def test_prevent_autograd():
    m = nn.Linear(3, 5)

    input = torch.randn((16, 3))
    output = m(input)

    m.weight.requires_grad_(False)
    m.bias.requires_grad_(False)

    output.backward(torch.ones_like(output), retain_graph=True)

    print(m.weight.grad)
    print(m.bias.grad)

    m.weight.requires_grad_(True)
    m.bias.requires_grad_(True)

    output.backward(torch.ones_like(output))

    print(m.weight.grad)
    print(m.bias.grad)
