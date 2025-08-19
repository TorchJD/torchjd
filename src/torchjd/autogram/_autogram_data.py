from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge
from torch.utils.hooks import RemovableHandle as TorchRemovableHandle

from ..aggregation import Weighting
from ._edge_registry import EdgeRegistry
from ._forward_hooks import ActivableHookFactory, ModuleHook
from ._gramian_accumulator import GramianAccumulator


class AutogramData:
    def __init__(
        self,
        modules: Iterable[nn.Module],
        weighting: Weighting,
    ):
        self._handles: list[TorchRemovableHandle] = []
        self._gramian_accumulator = GramianAccumulator()
        self._activable_hook_factory = ActivableHookFactory()
        self._target_edges = EdgeRegistry()
        self._weighting = weighting

        self._track_modules(modules)

    def _track_modules(self, modules: Iterable[nn.Module]) -> None:
        _modules = set(modules)

        # Add module forward hooks to compute jacobians
        for module in _modules:
            if next(module.parameters(recurse=False), None) is None:
                # Skip un-parameterized modules
                continue

            module_hook = self._activable_hook_factory.make_activable_hook(
                ModuleHook(self._target_edges, self._gramian_accumulator)
            )
            handle = module.register_forward_hook(module_hook)
            self._handles.append(handle)

    def untrack_modules(self) -> None:
        for handle in self._handles:
            handle.remove()

    def compute_gramian(self, output: Tensor, grad_outputs: Tensor | None = None) -> Tensor:
        if grad_outputs is None:
            grad_outputs = torch.ones_like(output)

        self._activable_hook_factory.deactivate()
        leaf_targets = list(self._target_edges.get_leaf_edges({get_gradient_edge(output)}, set()))

        _ = torch.autograd.grad(
            outputs=output,
            inputs=leaf_targets,
            grad_outputs=grad_outputs,
            retain_graph=True,
        )

        # If the gramian were None, then leaf_targets would be empty, so autograd.grad would
        # have failed. So gramian is necessarily a valid Tensor here.
        gramian = cast(Tensor, self._gramian_accumulator.gramian)

        # Reset everything that has a state
        self._gramian_accumulator.reset()
        self._activable_hook_factory.activate()
        self._target_edges.reset()

        return gramian

    def backward(self, tensor: Tensor, grad_outputs: Tensor | None = None) -> None:
        if tensor.ndim != 1:
            raise ValueError("Can use autogram.backward only on a vector of dimension 1.")
        gramian = self.compute_gramian(tensor, grad_outputs)
        weights = self._weighting(gramian)
        tensor.backward(weights)
