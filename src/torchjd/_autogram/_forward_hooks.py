from typing import Callable, cast

import torch
from torch import Tensor, nn
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten

from torchjd._autogram._autograd_functions import (
    _make_autogram_activator,
    _make_jacobian_accumulator,
)
from torchjd._autogram._gramian_accumulator import GramianAccumulator
from torchjd._autogram._hook_activator import HookActivator
from torchjd._autogram._target_registry import TargetRegistry
from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


def _get_module_hook(
    target_edges_registry: TargetRegistry,
    gramian_accumulator: GramianAccumulator,
    hook_activator: HookActivator,
) -> Callable[[nn.Module, PyTree, PyTree], PyTree]:
    def module_hook(module: nn.Module, args: PyTree, output: PyTree) -> PyTree:
        if not hook_activator.state:
            return output
        flat_outputs, tree_spec = tree_flatten(output, lambda x: isinstance(x, Tensor))

        if len(flat_outputs) == 0:
            # This can happen only if a module returns no Tensor, for instance some niche usage
            # such as a module that prints something.
            return output

        jacobian_accumulator = _make_jacobian_accumulator(
            module, gramian_accumulator, args, tree_spec
        )

        requires_grad_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        gramian_accumulator.track_parameter_paths(requires_grad_params)

        # We only care about running the JacobianAccumulator node, so we need one of its child
        # edges (the edges of the original ouputs of the model) as target. For memory
        # efficiency, we select the smallest one.
        numels = torch.tensor([t.numel() for t in flat_outputs])
        index = cast(int, numels.argmin().item())
        target_edges_registry.register(flat_outputs[index])

        return tree_unflatten(jacobian_accumulator.apply(*flat_outputs), tree_spec)

    return module_hook


def _get_model_hook(
    weighting: Weighting[PSDMatrix],
    target_edges_registry: TargetRegistry,
    gramian_accumulator: GramianAccumulator,
    hook_activator: HookActivator,
) -> Callable[[nn.Module, PyTree, PyTree], PyTree]:

    def model_hook(_, args: PyTree, output: PyTree) -> PyTree:
        if not hook_activator.state:
            return output

        input_tensors = [a for a in tree_flatten(args)[0] if isinstance(a, Tensor)]

        flat_outputs, tree_spec = tree_flatten(output)
        autogram_activator = _make_autogram_activator(
            flat_outputs,
            input_tensors,
            weighting,
            target_edges_registry,
            gramian_accumulator,
            hook_activator,
        )
        hook_activator.deactivate()
        activator_flat_outputs = autogram_activator.apply(*flat_outputs)
        return tree_unflatten(activator_flat_outputs, tree_spec)

    return model_hook
