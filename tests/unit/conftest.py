import os
import random as rand

import torch
from pytest import fixture

try:
    DEVICE = os.environ["PYTEST_TORCH_DEVICE"]
except KeyError:
    DEVICE = "cpu"  # Default to cpu if environment variable not set

if DEVICE != "cuda" and DEVICE != "cpu":
    raise ValueError(f"Invalid value of environment variable PYTEST_TORCH_DEVICE: {DEVICE}")

if DEVICE == "cuda" and not torch.cuda.is_available():
    raise ValueError('Requested device "cuda" but cuda is not available.')


@fixture(autouse=True)
def fix_randomness() -> None:
    rand.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
