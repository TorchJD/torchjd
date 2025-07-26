import time

import torch
from pytest import mark
from torch import Tensor, nn
from torch.nn import Flatten, ReLU
from torch.utils._pytree import PyTree
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
            nn.Sequential(nn.MaxPool2d(3), ReLU(), Flatten()),
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


class ModuleThatTakesString(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 10))
        self.matrix2 = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor, string: str):
        if string == "test":
            return input @ self.matrix1
        else:
            return input @ self.matrix2


class ModelThatTakesString(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = ModuleThatTakesString()

    def forward(self, input: Tensor):
        return self.module(input, "test") + self.module(input, "definitely not a test")


class MultiInputMultiOutputNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 60))
        self.matrix2 = nn.Parameter(torch.randn(50, 70))

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        input = sum(inputs)
        return input @ self.matrix1, input @ self.matrix2


class MultiInputNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 60))

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        input = sum(inputs)
        return input @ self.matrix1


class SingleInputSingleOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mimo = MultiInputMultiOutputNN()

    def forward(self, input: Tensor) -> Tensor:
        return torch.concatenate(list(self.mimo(input, input)), dim=1)


class SingleInputSingleOutputModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.miso = MultiInputNN()

    def forward(self, input: Tensor) -> Tensor:
        return self.miso(input, input)


class PyTreeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn(50, 50))
        self.matrix2 = nn.Parameter(torch.randn(50, 60))
        self.matrix3 = nn.Parameter(torch.randn(50, 70))
        self.matrix4 = nn.Parameter(torch.randn(50, 80))
        self.matrix5 = nn.Parameter(torch.randn(50, 90))

    def forward(self, input: Tensor) -> PyTree:
        return {
            "first": (input @ self.matrix1, [input @ self.matrix2, input @ self.matrix3]),
            "second": input @ self.matrix4,
            "third": ([(input @ self.matrix5,)],),
        }


class PyTreeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pytree_module = PyTreeModule()

    def forward(self, input: Tensor):
        first, second, third = self.pytree_module(input).values()
        output1, output23 = first
        output2, output3 = output23
        output4 = second
        output5 = third[0][0][0]

        return torch.concatenate([output1, output2, output3, output4, output5], dim=1)


class ModuleWithParameterReuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(50, 10))

    def forward(self, input: Tensor):
        return input @ self.matrix + input @ self.matrix


class MatMulModule(nn.Module):
    def __init__(self, matrix: nn.Parameter):
        super().__init__()
        self.matrix = matrix

    def forward(self, input: Tensor):
        return input @ self.matrix


class ModelWithInterModuleParameterReuse(nn.Module):
    def __init__(self):
        super().__init__()
        matrix = nn.Parameter(torch.randn(50, 10))
        self.module1 = MatMulModule(matrix)
        self.module2 = MatMulModule(matrix)

    def forward(self, input: Tensor):
        return self.module1(input) + self.module2(input)


class ModelWithModuleReuse(nn.Module):
    def __init__(self):
        super().__init__()
        matrix = nn.Parameter(torch.randn(50, 10))
        self.module = MatMulModule(matrix)

    def forward(self, input: Tensor):
        return self.module(input) + self.module(input)


class ModelWithFreeParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(15, 16))  # Free parameter
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(16, 50)
        self.linear2 = nn.Linear(50, 60)
        self.linear3 = nn.Linear(60, 70)
        self.linear4 = nn.Linear(70, 80)

    def forward(self, input: Tensor):
        output = self.relu(input @ self.matrix)
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.linear4(output)
        return output


class ModelWithNoFreeParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(15, 16, bias=False)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(16, 50)
        self.linear2 = nn.Linear(50, 60)
        self.linear3 = nn.Linear(60, 70)
        self.linear4 = nn.Linear(70, 80)

    def forward(self, input: Tensor):
        output = self.relu(self.linear0(input))
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.linear4(output)
        return output


@mark.parametrize(
    ["model", "single_input_shape"],
    [
        (Cifar10Model(), (3, 32, 32)),
        (FlatNonSequentialNN(), (9,)),
        (SingleInputSingleOutputModel(), (50,)),
        (SingleInputSingleOutputModel2(), (50,)),
        (PyTreeModel(), (50,)),
        (ModelWithFreeParameter(), (15,)),
        (ModelWithNoFreeParameter(), (15,)),
    ],
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

    def fn_autograd():
        autograd_forward_backward(model, criterion, input, target)

    def init_fn_autograd():
        torch.cuda.empty_cache()
        fn_autograd()

    def fn_autojac():
        autojac_forward_backward(model, criterion, input, target, A)

    def init_fn_autojac():
        torch.cuda.empty_cache()
        fn_autojac()

    def fn_autogram():
        autogram_forward_backward(model, criterion, input, target, W)

    def init_fn_autogram():
        torch.cuda.empty_cache()
        fn_autogram()

    def optionally_cuda_sync():
        if str(DEVICE).startswith("cuda"):
            torch.cuda.synchronize()

    def pre_fn():
        model.zero_grad()
        optionally_cuda_sync()

    def post_fn():
        optionally_cuda_sync()

    n_runs = 10
    autograd_times = torch.tensor(time_call(fn_autograd, init_fn_autograd, pre_fn, post_fn, n_runs))
    print(f"autograd times (avg = {autograd_times.mean():.5f}, std = {autograd_times.std():.5f}")
    print(autograd_times)
    print()

    autojac_times = torch.tensor(time_call(fn_autojac, init_fn_autojac, pre_fn, post_fn, n_runs))
    print(f"autojac times (avg = {autojac_times.mean():.5f}, std = {autojac_times.std():.5f}")
    print(autojac_times)
    print()

    autogram_times = torch.tensor(time_call(fn_autogram, init_fn_autogram, pre_fn, post_fn, n_runs))
    print(f"autogram times (avg = {autogram_times.mean():.5f}, std = {autogram_times.std():.5f}")
    print(autogram_times)
    print()


@mark.parametrize(
    ["model", "single_input_shape"],
    [
        (Cifar10Model(), (3, 32, 32)),
        (FlatNonSequentialNN(), (9,)),
        (SingleInputSingleOutputModel(), (50,)),
        (SingleInputSingleOutputModel2(), (50,)),
        (PyTreeModel(), (50,)),
        (ModuleWithParameterReuse(), (50,)),
        (ModelWithInterModuleParameterReuse(), (50,)),
        (ModelWithModuleReuse(), (50,)),
        (ModelWithFreeParameter(), (15,)),
        (ModelWithNoFreeParameter(), (15,)),
    ],
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


def noop():
    pass


def time_call(fn, init_fn=noop, pre_fn=noop, post_fn=noop, n_runs: int = 10) -> list[float]:
    init_fn()

    times = []
    for _ in range(n_runs):
        pre_fn()
        start = time.perf_counter()
        fn()
        post_fn()
        elapsed_time = time.perf_counter() - start
        times.append(elapsed_time)

    return times


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
