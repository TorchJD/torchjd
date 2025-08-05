from typing import Callable, cast

import torch
from torch import Tensor, nn
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten

from torchjd._autogram._activator import Activator
from torchjd._autogram._autograd_functions import _make_autogram_scaler, _make_jacobian_accumulator
from torchjd._autogram._edge_registry import EdgeRegistry
from torchjd._autogram._gramian_accumulator import GramianAccumulator
from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


def _make_module_hook(
    target_edges: EdgeRegistry,
    gramian_accumulator: GramianAccumulator,
    hook_activator: Activator,
) -> Callable[[nn.Module, PyTree, PyTree], PyTree]:
    def module_hook(module: nn.Module, args: PyTree, output: PyTree) -> PyTree:
        if not hook_activator.is_active:
            return output
        flat_outputs, tree_spec = tree_flatten(output)

        if output is None:
            # This can happen only if a module returns no Tensor, for instance some niche usage
            # such as a module that prints something.
            return output

        JacobianAccumulator = _make_jacobian_accumulator(
            module, gramian_accumulator, args, tree_spec
        )

        requires_grad_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        gramian_accumulator.track_parameter_paths(requires_grad_params)

        # We only care about running the JacobianAccumulator node, so we need one of its child
        # edges (the edges of the original ouputs of the model) as target. For memory
        # efficiency, we select the smallest one.
        numels = torch.tensor([t.numel() for t in flat_outputs])
        index = cast(int, numels.argmin().item())
        target_edges.register(flat_outputs[index])

        return tree_unflatten(JacobianAccumulator.apply(*flat_outputs), tree_spec)

    return module_hook


def _make_model_hook(
    weighting: Weighting[PSDMatrix],
    target_edges: EdgeRegistry,
    gramian_accumulator: GramianAccumulator,
    hook_activator: Activator,
) -> Callable[[nn.Module, PyTree, PyTree], PyTree]:

    def model_hook(_, args: PyTree, output: PyTree) -> PyTree:
        if not hook_activator.is_active:
            return output

        input_tensors = [a for a in tree_flatten(args)[0] if isinstance(a, Tensor)]

        flat_outputs, tree_spec = tree_flatten(output)
        AutogramScaler = _make_autogram_scaler(
            flat_outputs,
            input_tensors,
            weighting,
            target_edges,
            gramian_accumulator,
            hook_activator,
        )
        hook_activator.deactivate()
        return tree_unflatten(AutogramScaler.apply(*flat_outputs), tree_spec)

    return model_hook
