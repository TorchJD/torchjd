from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten, tree_unflatten

from torchjd.aggregation import PSDMatrix, Weighting
from torchjd.autogram._utils.edge_registry import EdgeRegistry
from torchjd.autogram._utils.gramian_accumulator import GramianAccumulator
from torchjd.autogram._utils.vjp import get_instance_wise_vjp

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


class ActivableHookFactory:
    """
    This class converts module hooks into hooks that can be activated or deactivated.
    """

    def __init__(self):
        self.is_active = True

    def activate(self) -> None:
        self.is_active = True

    def deactivate(self) -> None:
        self.is_active = False

    def make_activable_hook(
        self, hook: Callable[[nn.Module, PyTree, PyTree], PyTree]
    ) -> Callable[[nn.Module, PyTree, PyTree], PyTree]:
        def activable_hook(module: nn.Module, args: PyTree, output: PyTree):
            if not self.is_active:
                return output
            return hook(module, args, output)

        return activable_hook


class ModuleHook:
    """
    Create a forward hook used to insert Jacobian accumulation nodes into the backward graph.

    The hook injects a JacobianAccumulator function into the computation graph after the module,
    enabling Gramian computation during autogram's first backward pass.

    :param target_edges: Registry for tracking gradient edges that serve as targets for the first
        differentiation.
    :param gramian_accumulator: Accumulator for collecting the Jacobians into a Gramian.
    :returns: Forward hook for a submodule.
    """

    def __init__(
        self,
        target_edges: EdgeRegistry,
        gramian_accumulator: GramianAccumulator,
    ):
        self.target_edges = target_edges
        self.gramian_accumulator = gramian_accumulator

    def __call__(self, module: nn.Module, args: PyTree, output: PyTree) -> PyTree:
        flat_outputs, tree_spec = tree_flatten(output)

        if not any(isinstance(t, Tensor) for t in flat_outputs):
            # This can happen only if a module returns no Tensor, for instance some niche usage
            # such as a module that prints something.
            return output

        JacobianAccumulator = _make_jacobian_accumulator(
            module, self.gramian_accumulator, args, tree_spec
        )

        requires_grad_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        self.gramian_accumulator.track_parameter_paths(requires_grad_params)

        # We only care about running the JacobianAccumulator node, so we need one of its child
        # edges (the edges of the original ouputs of the model) as target. For memory
        # efficiency, we select the smallest one.
        numels = torch.tensor([t.numel() for t in flat_outputs])
        index = cast(int, numels.argmin().item())
        self.target_edges.register(get_gradient_edge(flat_outputs[index]))

        return tree_unflatten(JacobianAccumulator.apply(*flat_outputs), tree_spec)


class ModelHook:
    """
    Create a forward hook that inserts the autogram scaling node into the backward graph.

    The hook injects an AutogramScaler function at the model's output to coordinate autogram's
    two backward passes.
    """

    def __init__(
        self,
        weighting: Weighting[PSDMatrix],
        target_edges: EdgeRegistry,
        gramian_accumulator: GramianAccumulator,
        activable_hook_factory: ActivableHookFactory,
    ):
        self.weighting = weighting
        self.target_edges = target_edges
        self.gramian_accumulator = gramian_accumulator
        self.activable_hook_factory = activable_hook_factory

    def __call__(self, _, args: PyTree, output: PyTree) -> PyTree:
        input_tensors = [a for a in tree_flatten(args)[0] if isinstance(a, Tensor)]
        output_tensors = [a for a in tree_flatten(output)[0] if isinstance(a, Tensor)]

        flat_outputs, tree_spec = tree_flatten(output)
        AutogramScaler = _make_autogram_scaler(
            flat_outputs,
            input_tensors,
            output_tensors,
            self.weighting,
            self.target_edges,
            self.gramian_accumulator,
            self.activable_hook_factory,
        )
        self.activable_hook_factory.deactivate()
        return tree_unflatten(AutogramScaler.apply(*flat_outputs), tree_spec)


def _make_jacobian_accumulator(
    module: nn.Module,
    gramian_accumulator: GramianAccumulator,
    args: PyTree,
    tree_spec: TreeSpec,
) -> type[torch.autograd.Function]:

    class JacobianAccumulator(torch.autograd.Function):
        """
        Autograd function that accumulates Jacobian Gramians during the first backward pass.

        Acts as identity on forward pass. On the first backward pass of the autogram algorithm,
        computes the Jacobian of outputs w.r.t. module parameters and feeds it to the gramian
        accumulator. Uses a toggle mechanism to activate only during the first backward pass of the
        autogram algorithm.
        """

        activated = True

        @staticmethod
        def forward(*xs: Tensor) -> tuple[Tensor, ...]:
            return tuple([x.detach() for x in xs])

        @staticmethod
        def setup_context(*_):
            pass

        @staticmethod
        def backward(ctx, *flat_grad_outputs: Tensor):
            if not JacobianAccumulator.activated:
                JacobianAccumulator.activated = True
                return flat_grad_outputs

            JacobianAccumulator.activated = False

            grad_outputs = tree_unflatten(flat_grad_outputs, tree_spec)
            jacobians = torch.vmap(get_instance_wise_vjp(module))(grad_outputs, args)

            gramian_accumulator.accumulate_path_jacobians(
                {
                    module.get_parameter(param_name): jacobian
                    for param_name, jacobian in jacobians.items()
                }
            )

            return flat_grad_outputs

    return JacobianAccumulator


def _make_autogram_scaler(
    flat_outputs: PyTree,
    input_tensors: list[Tensor],
    output_tensors: list[Tensor],
    weighting: Weighting[PSDMatrix],
    target_edges: EdgeRegistry,
    gramian_accumulator: GramianAccumulator,
    activable_hook_factory: ActivableHookFactory,
) -> type[torch.autograd.Function]:

    excluded_edges = {get_gradient_edge(t) for t in input_tensors if t.requires_grad}
    roots = {get_gradient_edge(t) for t in output_tensors}
    leaf_targets = list(target_edges.get_leaf_edges(roots, excluded_edges))

    class AutogramScaler(torch.autograd.Function):
        """
        Autograd function that coordinates the autogram algorithm's two-phase backward pass.

        Triggers the first backward pass to accumulate the Gramian of the Jacobian, computes weights
        from the Gramian using the provided weighting, then scales gradients for the second backward
        pass.
        """

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

            # If the gramian were None, then leaf_targets would be empty, so autograd.grad would
            # have failed. So gramian is necessarily a valid Tensor here.
            gramian = cast(Tensor, gramian_accumulator.gramian)

            # Reset everything that has a state
            gramian_accumulator.reset()
            activable_hook_factory.activate()
            target_edges.reset()

            weights = weighting(gramian)
            scaled_grad_outputs = tuple(
                [torch.einsum("b...,b->b...", grad_output, weights) for grad_output in grad_outputs]
            )
            return scaled_grad_outputs

    return AutogramScaler
