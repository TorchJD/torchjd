import gc
import time

import torch
from device import DEVICE
from utils.architectures import (
    AlexNet,
    Cifar10Model,
    FreeParam,
    GroupNormMobileNetV3Small,
    InstanceNormMobileNetV2,
    InstanceNormResNet18,
    ModuleFactory,
    NoFreeParam,
    SqueezeNet,
    WithTransformerLarge,
    get_in_out_shapes,
)
from utils.forward_backwards import (
    autograd_forward_backward,
    autograd_gramian_forward_backward,
    autogram_forward_backward,
    autojac_forward_backward,
    make_mse_loss_fn,
)
from utils.tensors import make_tensors

from torchjd.aggregation import Mean
from torchjd.autogram import Engine

PARAMETRIZATIONS = [
    (ModuleFactory(WithTransformerLarge), 8),
    (ModuleFactory(FreeParam), 64),
    (ModuleFactory(NoFreeParam), 64),
    (ModuleFactory(Cifar10Model), 64),
    (ModuleFactory(AlexNet), 8),
    (ModuleFactory(InstanceNormResNet18), 16),
    (ModuleFactory(GroupNormMobileNetV3Small), 16),
    (ModuleFactory(SqueezeNet), 4),
    (ModuleFactory(InstanceNormMobileNetV2), 2),
]


def compare_autograd_autojac_and_autogram_speed(factory: ModuleFactory, batch_size: int):
    model = factory()
    input_shapes, output_shapes = get_in_out_shapes(model)
    inputs = make_tensors(batch_size, input_shapes)
    targets = make_tensors(batch_size, output_shapes)
    loss_fn = make_mse_loss_fn(targets)

    A = Mean()
    W = A.weighting

    print(
        f"\nTimes for forward + backward on {factory} with BS={batch_size}, A={A}" f" on {DEVICE}."
    )

    def fn_autograd():
        autograd_forward_backward(model, inputs, loss_fn)

    def init_fn_autograd():
        torch.cuda.empty_cache()
        gc.collect()
        fn_autograd()

    def fn_autograd_gramian():
        autograd_gramian_forward_backward(model, inputs, loss_fn, W)

    def init_fn_autograd_gramian():
        torch.cuda.empty_cache()
        gc.collect()
        fn_autograd_gramian()

    def fn_autojac():
        autojac_forward_backward(model, inputs, loss_fn, A)

    def init_fn_autojac():
        torch.cuda.empty_cache()
        gc.collect()
        fn_autojac()

    def fn_autogram():
        autogram_forward_backward(model, engine, W, inputs, loss_fn)

    def init_fn_autogram():
        torch.cuda.empty_cache()
        gc.collect()
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

    autograd_gramian_times = torch.tensor(
        time_call(fn_autograd_gramian, init_fn_autograd_gramian, pre_fn, post_fn, n_runs)
    )
    print(
        f"autograd gramian times (avg = {autograd_gramian_times.mean():.5f}, std = "
        f"{autograd_gramian_times.std():.5f}"
    )
    print(autograd_gramian_times)
    print()

    autojac_times = torch.tensor(time_call(fn_autojac, init_fn_autojac, pre_fn, post_fn, n_runs))
    print(f"autojac times (avg = {autojac_times.mean():.5f}, std = {autojac_times.std():.5f}")
    print(autojac_times)
    print()

    engine = Engine(model, batch_dim=0)
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
    for factory, batch_size in PARAMETRIZATIONS:
        compare_autograd_autojac_and_autogram_speed(factory, batch_size)
        print("\n")


if __name__ == "__main__":
    # To test this on cuda, add the following environment variables when running this:
    # CUBLAS_WORKSPACE_CONFIG=:4096:8;PYTEST_TORCH_DEVICE=cuda:0
    main()
