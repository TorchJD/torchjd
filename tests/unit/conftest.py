import gc
import os
import random as rand

import torch
from pytest import fixture, mark

try:
    _device_str = os.environ["PYTEST_TORCH_DEVICE"]
except KeyError:
    _device_str = "cpu"  # Default to cpu if environment variable not set

if _device_str != "cuda:0" and _device_str != "cpu":
    raise ValueError(f"Invalid value of environment variable PYTEST_TORCH_DEVICE: {_device_str}")

if _device_str == "cuda:0" and not torch.cuda.is_available():
    raise ValueError('Requested device "cuda:0" but cuda is not available.')

DEVICE = torch.device(_device_str)


@fixture(autouse=True)
def fix_randomness() -> None:
    rand.seed(0)
    torch.manual_seed(0)

    # Only force to use deterministic algorithms on CPU.
    # This is because the CI currently runs only on CPU, so we don't really need perfect
    # reproducibility on GPU. We also use GPU to benchmark algorithms, and we would rather have them
    # use non-deterministic but faster algorithms.
    if DEVICE.type == "cpu":
        torch.use_deterministic_algorithms(True)


@fixture(autouse=True)
def garbage_collect_if_marked(request):
    """
    Since garbage collection takes some time, we only do it when needed (when the test or the
    parametrization of the test is marked with mark.garbage_collect). This is currently useful for
    freeing CUDA memory after a lot has been allocated.
    """

    yield
    if request.node.get_closest_marker("garbage_collect"):
        gc.collect()


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "garbage_collect: do garbage collection after test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip_slow = mark.skip(reason="Slow test. Use --runslow to run it.")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
