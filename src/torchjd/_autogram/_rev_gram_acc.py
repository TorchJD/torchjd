from collections import Counter, deque
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.autograd.graph import GradientEdge, get_gradient_edge
from torch.nn import Parameter
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten, tree_map, tree_unflatten

from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


# TODO: document the cases where it doesn't work: non-batched operations, operations batched on
#  dim != 0 (rnns, transformers, ...), free nn.Parameter used before other nn.Modules (should work
#  but slowly).

# Second release: handle inputs that are not batched or that are batched on dim != 0
# TODO: test with RNN, Transformer
# TODO: add function mapping a module (and its args) to how to unbatch the args.
#  The output type could be tuple[Optional[int], ...], with one element per arg of the module


class GramianAccumulator:
    def __init__(self):
        self._gramian = None
        self.jacobians = dict()
        self.counter = Counter()

    def track_parameters(self, tensors: Iterable[Tensor]) -> None:
        self.counter.update(tensors)

    def add_jacobian(self, tensor: Tensor, jacobian: Tensor) -> None:
        if tensor in self.jacobians:
            self.jacobians[tensor] += jacobian
        else:
            self.jacobians[tensor] = jacobian
        self.counter.subtract([tensor])
        if self.counter[tensor] == 0:
            self._accumulate_jacobian(self.jacobians[tensor])
            del self.counter[tensor]
            del self.jacobians[tensor]

    def _accumulate_jacobian(self, jacobian: Tensor) -> None:
        jacobian_matrix = torch.flatten(jacobian, start_dim=1)
        if self._gramian is not None:
            self._gramian.addmm_(jacobian_matrix, jacobian_matrix.T)
        else:
            self._gramian = torch.mm(jacobian_matrix, jacobian_matrix.T)

    @property
    def gramian(self) -> Tensor:
        if len(self.counter) != 0 or len(self.jacobians) != 0:
            shape_to_count = [(k.shape, v) for k, v in self.counter.items()]
            raise ValueError(
                f"Some tracked parameters are still not at a count of 0, {shape_to_count}."
            )

        return self._gramian


def make_jacobian_accumulator(
    module: nn.Module,
    gramian_accumulator: GramianAccumulator,
    args: Any,
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

                def get_vjp(grad_outputs_j: PyTree, *inputs_j) -> tuple[Tensor, ...]:
                    # Note: we use unsqueeze(0) to turn a single activation (or grad_output) into a
                    # "batch" of 1 activation (or grad_output). This is because some layers (e.g.
                    # nn.Flatten) do not work equivalently if they're provided with a batch or with
                    # an element of a batch. We thus always provide them with batches, just of a
                    # different size.
                    inputs_j = [input_j.unsqueeze(0) for input_j in inputs_j]
                    grad_outputs_j = tree_map(lambda x: x.unsqueeze(0), grad_outputs_j)

                    # _vjp_from_module returns a function that computes the vjp w.r.t. to the
                    # primals (tuple), here the functional has a single primal which is
                    # dict(module.named_parameters()). We therefore take the 0'th element to obtain
                    # the dict of gradients w.r.t. the module's named_parameters.
                    return _vjp_from_module(module, *inputs_j)(grad_outputs_j)[0]

                grad_outputs = tree_unflatten(flat_grad_outputs, tree_spec)
                jacobians = torch.vmap(get_vjp)(grad_outputs, *args)

                for param_name, jacobian in jacobians.items():
                    gramian_accumulator.add_jacobian(module.get_parameter(param_name), jacobian)

            activated = not activated
            return flat_grad_outputs

    return JacobianAccumulator


class _ModelAugmenter:
    def __init__(self, model: nn.Module, weighting: Weighting[PSDMatrix]):
        self.model = model
        self.weighting = weighting
        self._handles = []

        self._gramian_accumulator = GramianAccumulator()
        self._are_hooks_activated = True
        self._target_edges_registry = []

    def augment(self):
        self._hook_submodules()
        self._hook_model()

    def unhook(self) -> None:
        for handle in self._handles:
            handle.remove()

    def _hook_submodules(self) -> None:
        for module in self.model.modules():
            if next(module.parameters(recurse=False), None) is None:
                # Skip un-parameterized modules
                continue
            self._hook_module(module)

    def _hook_module(self, module: nn.Module) -> None:
        def module_hook(_, args, output: PyTree) -> PyTree:
            if not self._are_hooks_activated:
                return output
            flat_outputs, tree_spec = tree_flatten(output)

            if len(flat_outputs) == 0:
                # This can happen only if a module returns no Tensor, for instance some niche usage such
                # as a module that prints something.
                return output

            jacobian_accumulator = make_jacobian_accumulator(
                module, self._gramian_accumulator, args, tree_spec
            )

            requires_grad_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            self._gramian_accumulator.track_parameters(requires_grad_params)

            # We only care about running the JacobianAccumulator node, so we need one of its child edges
            # (the edges of the original ouputs of the model) as target. For memory efficiency, we
            # select the smallest one.
            numels = torch.tensor([t.numel() for t in flat_outputs])
            index = numels.argmin().item()
            self._target_edges_registry.append(get_gradient_edge(flat_outputs[index]))

            return tree_unflatten(jacobian_accumulator.apply(*flat_outputs), tree_spec)

        self._handles.append(module.register_forward_hook(module_hook))

    def _hook_model(self) -> None:

        def model_hook(_, __, output: PyTree) -> PyTree:
            if not self._are_hooks_activated:
                return output

            leaf_targets = targets_to_leaf_targets(self._target_edges_registry, set())
            flat_outputs, tree_spec = tree_flatten(output)
            autogram_activator = self._make_autogram_activator(flat_outputs, leaf_targets)
            self._deactivate_module_hooks()
            activator_flat_outputs = autogram_activator.apply(*flat_outputs)
            return tree_unflatten(activator_flat_outputs, tree_spec)

        self._handles.append(self.model.register_forward_hook(model_hook))

    def _make_autogram_activator(
        self, flat_outputs: PyTree, leaf_targets: list[GradientEdge]
    ) -> type[torch.autograd.Function]:

        activated = True

        class AutogramActivator(torch.autograd.Function):
            @staticmethod
            def forward(*xs: Tensor) -> tuple[Tensor, ...]:
                return tuple([x.detach() for x in xs])

            @staticmethod
            def setup_context(*_):
                pass

            @staticmethod
            def backward(ctx, *grad_outputs: Tensor):
                nonlocal activated
                if not activated:
                    activated = True
                    return grad_outputs
                activated = False
                _ = torch.autograd.grad(
                    outputs=flat_outputs,
                    inputs=leaf_targets,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                )
                gramian = self._gramian_accumulator.gramian
                self._reset()
                weights = self.weighting(gramian).unsqueeze(1)
                scaled_grad_outputs = tuple([weights * grad_output for grad_output in grad_outputs])
                return scaled_grad_outputs

        return AutogramActivator

    def _reset(self):
        self._gramian_accumulator = GramianAccumulator()
        self._are_hooks_activated = True
        self._target_edges_registry = []

    def _deactivate_module_hooks(self) -> None:
        self._are_hooks_activated = False


class AutogramHandle:
    def __init__(self, manager: _ModelAugmenter):
        self._manager = manager

    def remove(self):
        self._manager.unhook()


def next_edges(edge: GradientEdge) -> list[GradientEdge]:
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]


def targets_to_leaf_targets(
    targets: list[GradientEdge], excluded: set[GradientEdge]
) -> list[GradientEdge]:
    targets = set(targets)
    nodes_to_traverse = deque((child, target) for target in targets for child in next_edges(target))

    already_added = {child for child, _ in nodes_to_traverse}

    while nodes_to_traverse:
        node, origin = nodes_to_traverse.popleft()
        if node in targets:
            excluded.add(origin)
        else:
            for child in next_edges(node):
                if child not in already_added:
                    nodes_to_traverse.append((child, origin))
                    already_added.add(child)

    return list(targets - excluded)


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


def _vjp_from_module(module: nn.Module, *inputs) -> Callable:
    named_params = dict(module.named_parameters(recurse=False))
    requires_grad_named_params = {k: v for k, v in named_params.items() if v.requires_grad}
    no_requires_grad_named_params = {k: v for k, v in named_params.items() if not v.requires_grad}

    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers()), **no_requires_grad_named_params}
        return torch.func.functional_call(module, all_state, tuple(inputs))

    return torch.func.vjp(functional_model_call, requires_grad_named_params)[1]
