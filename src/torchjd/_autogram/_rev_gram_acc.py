from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import GradientEdge, get_gradient_edge
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle

from torchjd._autogram._utils import GramianAccumulator, TargetRegistry, get_instance_wise_vjp
from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


def make_jacobian_accumulator(
    module: nn.Module,
    gramian_accumulator: GramianAccumulator,
    args: PyTree,
    tree_spec: TreeSpec,
) -> type[torch.autograd.Function]:

    activated = True

    class JacobianAccumulator(torch.autograd.Function):
        @staticmethod
        def forward(*xs: Tensor) -> tuple[Tensor, ...]:
            return tuple([x.detach() for x in xs])

        @staticmethod
        def setup_context(*_):
            pass

        @staticmethod
        def backward(ctx, *flat_grad_outputs: Tensor):
            nonlocal activated
            if activated:

                grad_outputs = tree_unflatten(flat_grad_outputs, tree_spec)
                jacobians = torch.vmap(get_instance_wise_vjp(module))(grad_outputs, args)

                gramian_accumulator.accumulate_path_jacobians(
                    {
                        module.get_parameter(param_name): jacobian
                        for param_name, jacobian in jacobians.items()
                    }
                )

            activated = not activated
            return flat_grad_outputs

    return JacobianAccumulator


class _ModelAugmenter:
    def __init__(self, model: nn.Module, weighting: Weighting[PSDMatrix]):
        self._model = model
        self._weighting = weighting
        self._handles: list[RemovableHandle] = []

        self._gramian_accumulator = GramianAccumulator()
        self._are_hooks_activated = True
        self._target_edges_registry = TargetRegistry()

    def augment(self):
        self._hook_submodules()
        self._hook_model()

    def unhook(self) -> None:
        for handle in self._handles:
            handle.remove()

    def _hook_submodules(self) -> None:
        for module in self._model.modules():
            if next(module.parameters(recurse=False), None) is None:
                # Skip un-parameterized modules
                continue
            self._hook_module(module)

    def _hook_module(self, module: nn.Module) -> None:
        def module_hook(_, args: PyTree, output: PyTree) -> PyTree:
            if not self._are_hooks_activated:
                return output
            flat_outputs, tree_spec = tree_flatten(output)

            if output is None:
                # This can happen only if a module returns no Tensor, for instance some niche usage
                # such as a module that prints something.
                return output

            jacobian_accumulator = make_jacobian_accumulator(
                module, self._gramian_accumulator, args, tree_spec
            )

            requires_grad_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            self._gramian_accumulator.track_parameter_paths(requires_grad_params)

            # We only care about running the JacobianAccumulator node, so we need one of its child
            # edges (the edges of the original ouputs of the model) as target. For memory
            # efficiency, we select the smallest one.
            numels = torch.tensor([t.numel() for t in flat_outputs])
            index = cast(int, numels.argmin().item())
            self._target_edges_registry.register(flat_outputs[index])

            return tree_unflatten(jacobian_accumulator.apply(*flat_outputs), tree_spec)

        self._handles.append(module.register_forward_hook(module_hook))

    def _hook_model(self) -> None:

        def model_hook(_, args: PyTree, output: PyTree) -> PyTree:
            if not self._are_hooks_activated:
                return output

            input_tensors = [a for a in tree_flatten(args)[0] if isinstance(a, Tensor)]
            excluded_edges = {get_gradient_edge(t) for t in input_tensors if t.requires_grad}
            leaf_targets = self._target_edges_registry.get_leaf_target_edges(excluded_edges)
            flat_outputs, tree_spec = tree_flatten(output)
            autogram_activator = self._make_autogram_activator(flat_outputs, leaf_targets)
            self._deactivate_module_hooks()
            activator_flat_outputs = autogram_activator.apply(*flat_outputs)
            return tree_unflatten(activator_flat_outputs, tree_spec)

        self._handles.append(self._model.register_forward_hook(model_hook))

    def _make_autogram_activator(
        self, flat_outputs: PyTree, leaf_targets: list[GradientEdge]
    ) -> type[torch.autograd.Function]:

        class AutogramActivator(torch.autograd.Function):
            @staticmethod
            def forward(*xs: Tensor) -> tuple[Tensor, ...]:
                return tuple([x.detach() for x in xs])

            @staticmethod
            def setup_context(*_):
                pass

            @staticmethod
            def backward(ctx, *grad_outputs: Tensor):
                _ = torch.autograd.grad(
                    outputs=flat_outputs,
                    inputs=leaf_targets,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                )
                gramian = self._gramian_accumulator.gramian
                self._reset()
                weights = self._weighting(gramian).unsqueeze(1)
                scaled_grad_outputs = tuple([weights * grad_output for grad_output in grad_outputs])
                return scaled_grad_outputs

        return AutogramActivator

    def _reset(self):
        self._gramian_accumulator = GramianAccumulator()
        self._are_hooks_activated = True
        self._target_edges_registry = TargetRegistry()

    def _deactivate_module_hooks(self) -> None:
        self._are_hooks_activated = False


class AutogramHandle:
    def __init__(self, manager: _ModelAugmenter):
        self._manager = manager

    def remove(self):
        self._manager.unhook()


def augment_model_with_iwrm_autogram(
    model: nn.Module,
    weighting: Weighting[PSDMatrix],
) -> AutogramHandle:
    """
    Usage:
    ```
    augment_model_with_iwrm_autogram(model, W)
    for input, target in ...:
        output = model(input)
        losses = criterion(output, target)
        losses.backward(torch.ones_like(losses))
    ```
    """
    model_augmenter = _ModelAugmenter(model, weighting)
    model_augmenter.augment()

    return AutogramHandle(model_augmenter)
