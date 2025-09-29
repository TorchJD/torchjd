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


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "xfail_if_cuda: mark test as xfail if running on cuda")


def pytest_collection_modifyitems(config, items):
    skip_slow = mark.skip(reason="Slow test. Use --runslow to run it.")
    xfail_cuda = mark.xfail(reason=f"Test expected to fail on {DEVICE}")
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "xfail_if_cuda" in item.keywords and str(DEVICE).startswith("cuda"):
            item.add_marker(xfail_cuda)
