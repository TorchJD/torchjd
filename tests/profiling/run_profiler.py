import gc
from pathlib import Path

import torch
from settings import DEVICE
from torch.profiler import ProfilerActivity, profile
from utils.architectures import (
    AlexNet,
    Cifar10Model,
    GroupNormMobileNetV3Small,
    InstanceNormMobileNetV2,
    InstanceNormResNet18,
    ModuleFactory,
    SqueezeNet,
    WithTransformerLarge,
)
from utils.forward_backwards import (
    autogram_forward_backward,
    autojac_forward_backward,
    make_mse_loss_fn,
)
from utils.tensors import make_inputs_and_targets

from torchjd.aggregation import Mean
from torchjd.aggregation._mean import MeanWeighting
from torchjd.autogram import Engine

PARAMETRIZATIONS = [
    (ModuleFactory(WithTransformerLarge), 4),
    (ModuleFactory(Cifar10Model), 64),
    (ModuleFactory(AlexNet), 4),
    (ModuleFactory(InstanceNormResNet18), 4),
    (ModuleFactory(GroupNormMobileNetV3Small), 8),
    (ModuleFactory(SqueezeNet), 4),
    (ModuleFactory(InstanceNormMobileNetV2), 2),
]


def main():
    for factory, batch_size in PARAMETRIZATIONS:
        profile_autojac(factory, batch_size)
        print("\n" + "=" * 80 + "\n")
        profile_autogram(factory, batch_size)
        print("\n" + "=" * 80 + "\n")


def profile_autojac(factory: ModuleFactory, batch_size: int) -> None:
    """
    Profiles memory and computation time of autojac forward and backward pass for a given
    architecture.

    Prints the result and saves it in the traces folder. The saved traces be viewed using chrome at
    chrome://tracing.

    :param factory: A ModuleFactory that creates the model to profile.
    :param batch_size: The batch size to use for profiling.
    """

    print(f"autojac: {factory} with batch_size={batch_size} on {DEVICE}:")

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model = factory()
    inputs, targets = make_inputs_and_targets(model, batch_size)
    loss_fn = make_mse_loss_fn(targets)
    aggregator = Mean()

    activities = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Warmup run
    autojac_forward_backward(model, inputs, loss_fn, aggregator)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Profiled run
    model.zero_grad()
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        autojac_forward_backward(model, inputs, loss_fn, aggregator)

    filename = f"{factory}-bs{batch_size}-{DEVICE.type}.json"
    torchjd_dir = Path(__file__).parent.parent.parent
    traces_dir = torchjd_dir / "traces" / "autojac"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / filename

    prof.export_chrome_trace(str(trace_path))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))


def profile_autogram(factory: ModuleFactory, batch_size: int) -> None:
    """
    Profiles memory and computation time of autogram forward and backward pass for a given
    architecture.

    Prints the result and saves it in the traces folder. The saved traces be viewed using chrome at
    chrome://tracing.

    :param factory: A ModuleFactory that creates the model to profile.
    :param batch_size: The batch size to use for profiling.
    """

    print(f"autogram: {factory} with batch_size={batch_size} on {DEVICE}:")

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model = factory()
    inputs, targets = make_inputs_and_targets(model, batch_size)
    loss_fn = make_mse_loss_fn(targets)
    engine = Engine(model, batch_dim=0)
    weighting = MeanWeighting()

    activities = [ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Warmup run
    autogram_forward_backward(model, inputs, loss_fn, engine, weighting)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Profiled run
    model.zero_grad()
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        autogram_forward_backward(model, inputs, loss_fn, engine, weighting)

    filename = f"{factory}-bs{batch_size}-{DEVICE.type}.json"
    torchjd_dir = Path(__file__).parent.parent.parent
    traces_dir = torchjd_dir / "traces" / "autogram"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / filename

    prof.export_chrome_trace(str(trace_path))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))


if __name__ == "__main__":
    # To test this on cuda, add the following environment variables when running this:
    # CUBLAS_WORKSPACE_CONFIG=:4096:8;PYTEST_TORCH_DEVICE=cuda:0
    main()
