import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge
from torch.utils._pytree import PyTree, TreeSpec, tree_unflatten

from torchjd._autogram._gramian_accumulator import GramianAccumulator
from torchjd._autogram._hook_activator import HookActivator
from torchjd._autogram._target_registry import TargetRegistry
from torchjd._autogram._vjp import get_instance_wise_vjp
from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting


def _make_jacobian_accumulator(
    module: nn.Module,
    gramian_accumulator: GramianAccumulator,
    args: PyTree,
    tree_spec: TreeSpec,
) -> type[torch.autograd.Function]:

    class JacobianAccumulator(torch.autograd.Function):

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


def _make_autogram_activator(
    flat_outputs: PyTree,
    input_tensors: list[Tensor],
    weighting: Weighting[PSDMatrix],
    target_edges_registry: TargetRegistry,
    gramian_accumulator: GramianAccumulator,
    hook_activator: HookActivator,
) -> type[torch.autograd.Function]:

    excluded_edges = {get_gradient_edge(t) for t in input_tensors if t.requires_grad}
    leaf_targets = target_edges_registry.get_leaf_target_edges(excluded_edges)

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

            gramian = gramian_accumulator.gramian

            # Should never happen, these asserts are temporary for development safety reason.
            assert len(gramian_accumulator._path_counter) == 0
            assert len(gramian_accumulator._summed_jacobians) == 0
            assert gramian is not None

            # Reset everything that has a state
            gramian_accumulator.reset()
            hook_activator.activate()
            target_edges_registry.reset()

            weights = weighting(gramian).unsqueeze(1)
            scaled_grad_outputs = tuple([weights * grad_output for grad_output in grad_outputs])
            return scaled_grad_outputs

    return AutogramActivator
