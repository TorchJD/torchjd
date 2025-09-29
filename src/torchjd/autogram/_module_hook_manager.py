import weakref
from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten, tree_map, tree_unflatten
from torch.utils.hooks import RemovableHandle as TorchRemovableHandle

from ._edge_registry import EdgeRegistry
from ._gramian_accumulator import GramianAccumulator
from ._vjp import VJP, AutogradVJP, FunctionalVJP, Vmapped

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
        has_batch_dim: bool,
    ):
        self._target_edges = target_edges
        self._gramian_accumulator = gramian_accumulator
        self._has_batch_dim = has_batch_dim
        self.gramian_accumulation_phase = BoolRef(False)
        self._handles: list[TorchRemovableHandle] = []

        # When the ModuleHookManager is not referenced anymore, there is no reason to keep the hooks
        # alive. In fact, keeping the hooks alive would also keep the target edges alive, which
        # would keep the graph or part of the graph alive. Since the graph contains nodes that store
        # the module in their context, which themselves reference their hooks, the hooks will be
        # caught in a reference cycle and will not be freed by the garbage collector. It is thus
        # important to remove the hooks whenever we're sure we won't need them anymore.
        # We could have used a __del__ method here, with the same effects, but weakref.finalize
        # seems to be a better practice (and it only works if the function to call is static).
        self._finalizer = weakref.finalize(self, ModuleHookManager.remove_hooks, self._handles)

    def hook_module(self, module: nn.Module) -> None:
        """
        Add a module hook used to insert Jacobian accumulation nodes into the backward graph.

        The hook injects a JacobianAccumulator function into the computation graph after the module,
        enabling Gramian computation.
        """

        hook = Hook(
            self.gramian_accumulation_phase,
            self._target_edges,
            self._gramian_accumulator,
            self._has_batch_dim,
        )
        self._handles.append(module.register_forward_hook(hook))

    @staticmethod
    def remove_hooks(handles: list[TorchRemovableHandle]) -> None:
        """
        Remove all registered hooks. This method is deliberately static so that it can be called by
        weakref.finalize.
        """

        for handle in handles:
            handle.remove()


class AccumulateJacobian(torch.autograd.Function):

    @staticmethod
    def forward(
        output_spec: TreeSpec,
        vjp: VJP,
        args: PyTree,
        gramian_accumulator: GramianAccumulator,
        module: nn.Module,
        *flat_grad_outputs: Tensor,
    ) -> None:
        # There is no non-batched dimension
        grad_outputs = tree_unflatten(flat_grad_outputs, output_spec)
        generalized_jacobians = vjp(grad_outputs, args)
        path_jacobians = AccumulateJacobian._make_path_jacobians(module, generalized_jacobians)
        gramian_accumulator.accumulate_path_jacobians(path_jacobians)

    @staticmethod
    def vmap(
        _,
        in_dims: PyTree,
        output_spec: TreeSpec,
        vjp: VJP,
        args: PyTree,
        gramian_accumulator: GramianAccumulator,
        module: nn.Module,
        *flat_jac_outputs: Tensor,
    ) -> tuple[None, None]:
        # There is a non-batched dimension
        jac_outputs = tree_unflatten(flat_jac_outputs, output_spec)
        # We do not vmap over the args for the non-batched dimension
        in_dims = (tree_unflatten(in_dims[5:], output_spec), tree_map(lambda _: None, args))
        generalized_jacobians = torch.vmap(vjp, in_dims=in_dims)(jac_outputs, args)
        path_jacobians = AccumulateJacobian._make_path_jacobians(module, generalized_jacobians)
        gramian_accumulator.accumulate_path_jacobians(path_jacobians)
        return None, None

    @staticmethod
    def _make_path_jacobians(
        module: nn.Module,
        generalized_jacobians: dict[str, Tensor],
    ) -> dict[Tensor, Tensor]:
        path_jacobians: dict[Tensor, Tensor] = {}
        for param_name, generalized_jacobian in generalized_jacobians.items():
            key = module.get_parameter(param_name)
            jacobian = generalized_jacobian.reshape([-1] + list(key.shape))
            path_jacobians[key] = jacobian
        return path_jacobians

    @staticmethod
    def setup_context(*_):
        pass


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
        gramian_accumulation_phase: BoolRef,
        output_spec: TreeSpec,
        vjp: VJP,
        args: PyTree,
        gramian_accumulator: GramianAccumulator,
        module: nn.Module,
        *xs: Tensor,
    ) -> tuple[Tensor, ...]:
        return tuple(x.detach() for x in xs)

    # For Python version > 3.10, the type of `inputs` should become
    # tuple[BoolRef, TreeSpec, VJP, PyTree, GramianAccumulator, nn.Module, *tuple[Tensor, ...]]
    @staticmethod
    def setup_context(
        ctx,
        inputs: tuple,
        _,
    ):
        ctx.gramian_accumulation_phase = inputs[0]
        ctx.output_spec = inputs[1]
        ctx.vjp = inputs[2]
        ctx.args = inputs[3]
        ctx.gramian_accumulator = inputs[4]
        ctx.module = inputs[5]

    @staticmethod
    def backward(ctx, *flat_grad_outputs: Tensor):
        if not ctx.gramian_accumulation_phase:
            return None, None, None, None, None, None, *flat_grad_outputs

        AccumulateJacobian.apply(
            ctx.output_spec,
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
        has_batch_dim: bool,
    ):
        self.gramian_accumulation_phase = gramian_accumulation_phase
        self.target_edges = target_edges
        self.gramian_accumulator = gramian_accumulator
        self.has_batch_dim = has_batch_dim

    def __call__(self, module: nn.Module, args: PyTree, output: PyTree) -> PyTree:
        if self.gramian_accumulation_phase:
            return output

        flat_outputs, output_spec = tree_flatten(output)

        if not any(isinstance(t, Tensor) and t.requires_grad for t in flat_outputs):
            # This can happen only if a module has a trainable param but outputs no tensor that
            # require grad
            return output

        requires_grad_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        self.gramian_accumulator.track_parameter_paths(requires_grad_params)

        # We only care about running the JacobianAccumulator node, so we need one of its child
        # edges (the edges of the original outputs of the model) as target. For memory
        # efficiency, we select the smallest one (that requires grad).
        inf = float("inf")
        preference = torch.tensor([t.numel() if t.requires_grad else inf for t in flat_outputs])
        index = cast(int, preference.argmin().item())
        self.target_edges.register(get_gradient_edge(flat_outputs[index]))

        vjp: VJP
        if self.has_batch_dim:
            vjp = Vmapped(FunctionalVJP(module))
        else:
            vjp = AutogradVJP(module, flat_outputs)

        autograd_fn_outputs = JacobianAccumulator.apply(
            self.gramian_accumulation_phase,
            output_spec,
            vjp,
            args,
            self.gramian_accumulator,
            module,
            *flat_outputs,
        )

        return tree_unflatten(autograd_fn_outputs, output_spec)
