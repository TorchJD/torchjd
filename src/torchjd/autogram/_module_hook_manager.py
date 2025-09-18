from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle as TorchRemovableHandle

from ._edge_registry import EdgeRegistry
from ._gramian_accumulator import GramianAccumulator
from ._vjp import get_functional_vjp

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


class BoolRef:
    """Class wrapping a boolean value, acting as a reference to this boolean value."""

    def __init__(self, value: bool):
        self.value = value

    def __bool__(self) -> bool:
        return self.value


class ModuleHookManager:
    """
    Class responsible for handling hooks and Nodes that computes the Gramian reverse accumulation.

    :param target_edges: Registry for tracking gradient edges that serve as targets for the first
        differentiation.
    :param gramian_accumulator: Accumulator for collecting the Jacobians into a Gramian.
    """

    def __init__(
        self,
        target_edges: EdgeRegistry,
        gramian_accumulator: GramianAccumulator,
    ):
        self._target_edges = target_edges
        self._gramian_accumulator = gramian_accumulator
        self.gramian_accumulation_phase = BoolRef(False)
        self._handles: list[TorchRemovableHandle] = []

    def hook_module(self, module: nn.Module) -> None:
        """
        Add a module hook used to insert Jacobian accumulation nodes into the backward graph.

        The hook injects a JacobianAccumulator function into the computation graph after the module,
        enabling Gramian computation.
        """

        hook = Hook(self.gramian_accumulation_phase, self._target_edges, self._gramian_accumulator)
        self._handles.append(module.register_forward_hook(hook))


class AccumulateJacobian(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, tree_spec: TreeSpec, vjp, args, gramian_accumulator, module, *flat_grad_outputs: Tensor
    ) -> None:
        grad_outputs = tree_unflatten(flat_grad_outputs, tree_spec)
        jacobians = vjp(grad_outputs, args)
        gramian_accumulator.accumulate_path_jacobians(
            {
                module.get_parameter(param_name): jacobian
                for param_name, jacobian in jacobians.items()
            }
        )


class JacobianAccumulator(torch.autograd.Function):
    """
    Autograd function that accumulates Jacobian Gramians during the first backward pass.

    Acts as identity on forward pass. During the autogram algorithm, computes the Jacobian
    of outputs w.r.t. module parameters and feeds it to the gramian accumulator. Uses a
    toggle mechanism to activate only during the Gramian accumulation phase.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        ctx,
        gramian_accumulation_phase: BoolRef,
        tree_spec,
        vjp,
        args,
        gramian_accumulator,
        module,
        *xs: Tensor,
    ) -> tuple[Tensor, ...]:
        ctx.gramian_accumulation_phase = gramian_accumulation_phase
        ctx.tree_spec = tree_spec
        ctx.vjp = vjp
        ctx.args = args
        ctx.gramian_accumulator = gramian_accumulator
        ctx.module = module
        return tuple([x.detach() for x in xs])

    @staticmethod
    def backward(ctx, *flat_grad_outputs: Tensor):
        if not ctx.gramian_accumulation_phase:
            return None, None, None, None, None, None, *flat_grad_outputs

        AccumulateJacobian.apply(
            ctx.tree_spec,
            ctx.vjp,
            ctx.args,
            ctx.gramian_accumulator,
            ctx.module,
            *flat_grad_outputs,
        )

        return None, None, None, None, None, None, *flat_grad_outputs


class Hook:
    def __init__(
        self,
        gramian_accumulation_phase: BoolRef,
        target_edges: EdgeRegistry,
        gramian_accumulator: GramianAccumulator,
    ):
        self.gramian_accumulation_phase = gramian_accumulation_phase
        self.target_edges = target_edges
        self.gramian_accumulator = gramian_accumulator

    def __call__(self, module: nn.Module, args: PyTree, output: PyTree) -> PyTree:
        if self.gramian_accumulation_phase:
            return output

        flat_outputs, tree_spec = tree_flatten(output)

        if not any(isinstance(t, Tensor) for t in flat_outputs):
            # This can happen only if a module returns no Tensor, for instance some niche usage
            # such as a module that prints something.
            return output

        requires_grad_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        self.gramian_accumulator.track_parameter_paths(requires_grad_params)

        # We only care about running the JacobianAccumulator node, so we need one of its child
        # edges (the edges of the original ouputs of the model) as target. For memory
        # efficiency, we select the smallest one (that requires grad).
        inf = float("inf")
        preference = torch.tensor([t.numel() if t.requires_grad else inf for t in flat_outputs])
        index = cast(int, preference.argmin().item())
        self.target_edges.register(get_gradient_edge(flat_outputs[index]))

        vjp = torch.vmap(get_functional_vjp(module))

        return tree_unflatten(
            JacobianAccumulator.apply(
                self.gramian_accumulation_phase,
                tree_spec,
                vjp,
                args,
                self.gramian_accumulator,
                module,
                *flat_outputs,
            ),
            tree_spec,
        )
