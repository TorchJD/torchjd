import random as rand

import pytest
import torch


@pytest.fixture(autouse=True)
def fix_randomness():
    rand.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
