import os
import random as rand

import torch
from pytest import fixture

try:
    _device_str = os.environ["PYTEST_TORCH_DEVICE"]
except KeyError:
    _device_str = "cpu"  # Default to cpu if environment variable not set

if _device_str != "cuda:0" and _device_str != "cpu":
    raise ValueError(f"Invalid value of environment variable PYTEST_TORCH_DEVICE: {_device_str}")

if _device_str == "cuda:0" and not torch.cuda.is_available():
    raise ValueError('Requested device "cuda:0" but cuda is not available.')

DEVICE = torch.device(_device_str)
torch.set_default_device(DEVICE)


@fixture(autouse=True)
def fix_randomness() -> None:
    rand.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
