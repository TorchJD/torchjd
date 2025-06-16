from contextlib import AbstractContextManager
from functools import partial
from typing import TypeAlias

import torch
from unit.conftest import DEVICE

ExceptionContext: TypeAlias = AbstractContextManager[Exception | None]

# Curried calls to torch functions that require a device so that we automatically fix the device
# for code written in the tests, while not affecting code written in src (what
# torch.set_default_device or what a too large `with torch.device(DEVICE)` context would have done).

empty_ = partial(torch.empty, device=DEVICE)
eye_ = partial(torch.eye, device=DEVICE)
ones_ = partial(torch.ones, device=DEVICE)
rand_ = partial(torch.rand, device=DEVICE)
randint_ = partial(torch.randint, device=DEVICE)
randn_ = partial(torch.randn, device=DEVICE)
randperm_ = partial(torch.randperm, device=DEVICE)
tensor_ = partial(torch.tensor, device=DEVICE)
zeros_ = partial(torch.zeros, device=DEVICE)
