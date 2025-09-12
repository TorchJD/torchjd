import time

import torch
from unit.conftest import DEVICE
from utils.architectures import (
    AlexNet,
    Cifar10Model,
    FreeParam,
    GroupNormMobileNetV3Small,
    InstanceNormMobileNetV2,
    InstanceNormResNet18,
    NoFreeParam,
    ShapedModule,
    SqueezeNet,
)
from utils.forward_backwards import (
    autograd_forward_backward,
    autograd_gramian_forward_backward,
    autogram_forward_backward,
    make_mse_loss_fn,
)
from utils.tensors import make_tensors

from torchjd.aggregation import Mean
from torchjd.autogram import Engine

PARAMETRIZATIONS = [
    (FreeParam, 64),
    (NoFreeParam, 64),
    (Cifar10Model, 64),
    (AlexNet, 8),
    (InstanceNormResNet18, 16),
    (GroupNormMobileNetV3Small, 16),
    (SqueezeNet, 16),
    (InstanceNormMobileNetV2, 8),
]


def compare_autograd_autojac_and_autogram_speed(architecture: type[ShapedModule], batch_size: int):
    input_shapes = architecture.INPUT_SHAPES
    output_shapes = architecture.OUTPUT_SHAPES
    inputs = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    model = architecture().to(device=DEVICE)

    A = Mean()
    W = A.weighting

    print(
        f"\nTimes for forward + backward on {architecture.__name__} with BS={batch_size}, A={A}"
        f" on {DEVICE}."
    )

    def fn_autograd():
        autograd_forward_backward(model, inputs, loss_fn)

    def init_fn_autograd():
        torch.cuda.empty_cache()
        fn_autograd()

    def fn_autojac():
        autograd_gramian_forward_backward(model, inputs, loss_fn, A)

    def init_fn_autojac():
        torch.cuda.empty_cache()
        fn_autojac()

    def fn_autogram():
        autogram_forward_backward(model, engine, W, inputs, loss_fn)

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

    engine = Engine(model.modules())
    autogram_times = torch.tensor(time_call(fn_autogram, init_fn_autogram, pre_fn, post_fn, n_runs))
    print(f"autogram times (avg = {autogram_times.mean():.5f}, std = {autogram_times.std():.5f}")
    print(autogram_times)
    print()


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


def main():
    for architecture, batch_size in PARAMETRIZATIONS:
        compare_autograd_autojac_and_autogram_speed(architecture, batch_size)
        print("\n")


if __name__ == "__main__":
    # To test this on cuda, add the following environment variables when running this:
    # CUBLAS_WORKSPACE_CONFIG=:4096:8;PYTEST_TORCH_DEVICE=cuda:0
    main()
