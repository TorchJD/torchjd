import random as rand
from contextlib import nullcontext
from importlib.util import find_spec

import torch
from pytest import RaisesExc, fixture, mark
from settings import DEVICE
from torch import Tensor
from utils.architectures import ModuleFactory

from torchjd.aggregation import Aggregator, Weighting


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
    config.addinivalue_line(
        "markers", "xfail_if_cagrad_not_installed: mark test as xfail if CAGrad is not installed"
    )
    config.addinivalue_line(
        "markers",
        "xfail_if_nashmtl_not_installed: mark test as xfail if NashMTL is not installed",
    )


def pytest_collection_modifyitems(config, items):
    skip_slow = mark.skip(reason="Slow test. Use --runslow to run it.")
    xfail_cuda = mark.xfail(reason=f"Test expected to fail on {DEVICE}")

    # Check if optional dependencies are installed
    cagrad_installed = all(find_spec(name) is not None for name in ["cvxpy", "clarabel"])
    nashmtl_installed = all(find_spec(name) is not None for name in ["cvxpy", "ecos"])

    xfail_cagrad = mark.xfail(reason="CAGrad dependencies not installed")
    xfail_nashmtl = mark.xfail(reason="NashMTL dependencies not installed")

    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "xfail_if_cuda" in item.keywords and str(DEVICE).startswith("cuda"):
            item.add_marker(xfail_cuda)
        if "xfail_if_cagrad_not_installed" in item.keywords and not cagrad_installed:
            item.add_marker(xfail_cagrad)
        if "xfail_if_nashmtl_not_installed" in item.keywords and not nashmtl_installed:
            item.add_marker(xfail_nashmtl)


def pytest_make_parametrize_id(config, val, argname):
    MAX_SIZE = 40
    optional_string = None  # Returning None means using pytest's way of making the string

    if isinstance(val, (Aggregator, ModuleFactory, Weighting)):
        optional_string = str(val)
    elif isinstance(val, Tensor):
        optional_string = "T" + str(list(val.shape))  # T to indicate that it's a tensor
    elif isinstance(val, (tuple, list, set)) and len(val) < 20:
        optional_string = str(val)
    elif isinstance(val, RaisesExc):
        optional_string = " or ".join([f"{exc.__name__}" for exc in val.expected_exceptions])
    elif isinstance(val, nullcontext):
        optional_string = "does_not_raise()"

    if isinstance(optional_string, str) and len(optional_string) > MAX_SIZE:
        optional_string = optional_string[: MAX_SIZE - 3] + "+++"  # Can't use dots with pytest

    return optional_string
