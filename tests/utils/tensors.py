from functools import partial

import torch
from torch import nn
from torch.utils._pytree import PyTree, tree_map

from tests.device import DEVICE
from tests.utils.architectures import get_in_out_shapes
from tests.utils.contexts import fork_rng

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


def make_inputs_and_targets(model: nn.Module, batch_size: int) -> tuple[PyTree, PyTree]:
    input_shapes, output_shapes = get_in_out_shapes(model)
    with fork_rng(seed=0):
        inputs = _make_tensors(batch_size, input_shapes)
        targets = _make_tensors(batch_size, output_shapes)

    return inputs, targets


def _make_tensors(batch_size: int, tensor_shapes: PyTree) -> PyTree:
    def is_leaf(s):
        return isinstance(s, tuple) and all([isinstance(e, int) for e in s])

    return tree_map(lambda s: randn_((batch_size,) + s), tensor_shapes, is_leaf=is_leaf)
