from functools import partial

import torch
from device import DEVICE
from torch.utils._pytree import PyTree, tree_map

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


def make_tensors(batch_size: int, tensor_shapes: PyTree) -> PyTree:
    def is_leaf(s):
        return isinstance(s, tuple) and all([isinstance(e, int) for e in s])

    return tree_map(lambda s: randn_((batch_size,) + s), tensor_shapes, is_leaf=is_leaf)
