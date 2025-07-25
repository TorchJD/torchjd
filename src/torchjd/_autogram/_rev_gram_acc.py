from collections import Counter, deque
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.autograd.graph import GradientEdge, get_gradient_edge
from torch.nn import Parameter
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten, tree_map, tree_unflatten

from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting

# TODO: add test with free nn.Parameter used before other nn.Modules
# TODO: document the cases where it doesn't work: non-batched operations, operations batched on
#  dim != 0 (rnns, transformers, ...), free nn.Parameter used before other nn.Modules (should work
#  but slowly).

# Second release: handle inputs that are not batched or that are batched on dim != 0
# TODO: test with RNN, Transformer
# TODO: add function mapping a module (and its args) to how to unbatch the args.
#  The output type could be tuple[Optional[int], ...], with one element per arg of the module


class GramianAccumulator:
    def __init__(self):
        self.gramian = None
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
        if self.gramian is not None:
            self.gramian.addmm_(jacobian_matrix, jacobian_matrix.T)
        else:
            self.gramian = torch.mm(jacobian_matrix, jacobian_matrix.T)


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
                activated = False

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
                for param_name, param in module.named_parameters(recurse=False):
                    gramian_accumulator.add_jacobian(param, jacobians[param_name])

            return flat_grad_outputs

    return JacobianAccumulator


def get_model_hook(
    gramian_accumulator: GramianAccumulator,
    target_edges_registry: list[GradientEdge],
) -> Callable:
    def forward_post_hook(module, args, output: PyTree) -> PyTree:

        flat_outputs, tree_spec = tree_flatten(output)

        if len(flat_outputs) == 0:
            # This can happen only if a module returns no Tensor, for instance some niche usage such
            # as a module that prints something.
            return output

        jacobian_accumulator = make_jacobian_accumulator(
            module, gramian_accumulator, args, tree_spec
        )

        gramian_accumulator.track_parameters(module.parameters(recurse=False))

        # We only care about running the JacobianAccumulator node, so we need one of its child edges
        # (the edges of the original ouputs of the model) as target. For memory efficiency, we
        # select the smallest one.
        numels = torch.tensor([t.numel() for t in flat_outputs])
        index = numels.argmin().item()
        target_edges_registry.append(get_gradient_edge(flat_outputs[index]))

        return tree_unflatten(jacobian_accumulator.apply(*flat_outputs), tree_spec)

    return forward_post_hook


def augment_model(
    model: nn.Module,
    gramian_accumulator: GramianAccumulator,
    target_edges_registry: list[GradientEdge],
    forward_hook_handles: list,
):
    for module in model.modules():
        param_iterator = module.parameters(recurse=False)
        if any(True for _ in param_iterator):  # if there is at least one parameter
            forward_post_hook = get_model_hook(gramian_accumulator, target_edges_registry)
            forward_hook_handles.append(module.register_forward_hook(forward_post_hook))


def next_edges(edge: GradientEdge) -> list[GradientEdge]:
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]


def targets_to_leaf_targets(targets: list[GradientEdge]) -> list[GradientEdge]:
    targets = set(targets)
    nodes_to_traverse = deque((child, target) for target in targets for child in next_edges(target))

    already_added = {child for child, _ in nodes_to_traverse}
    traversed_targets = set()

    while nodes_to_traverse:
        node, origin = nodes_to_traverse.popleft()
        if node in targets:
            traversed_targets.add(origin)

        for child in next_edges(node):
            if child not in already_added:
                nodes_to_traverse.append((child, origin))
                already_added.add(child)

    return list(targets - traversed_targets)


def autogram_forward_backward(
    model: nn.Module,
    criterion: Callable,
    input: Tensor,
    target: Tensor,
    weighting: Weighting[PSDMatrix],
) -> tuple[Tensor, Tensor, Tensor]:
    target_edges_registry = []
    forward_hook_handles = []
    gramian_accumulator = GramianAccumulator()
    augment_model(model, gramian_accumulator, target_edges_registry, forward_hook_handles)

    output = model(input)

    losses = criterion(output, target)

    for handle in forward_hook_handles:
        handle.remove()

    leaf_targets = targets_to_leaf_targets(target_edges_registry)

    # Note: grad_outputs doesn't really matter here. The purpose of this is to compute the required
    # grad_outputs and trigger the tensor hooks with them
    _ = torch.autograd.grad(
        outputs=losses,
        inputs=leaf_targets,
        grad_outputs=torch.ones_like(losses),
        retain_graph=True,
    )

    gramian = gramian_accumulator.gramian
    weights = weighting(gramian)

    weighted_loss = losses @ weights
    weighted_loss.backward()

    return output, losses, weights


def autogram_forward_backward_old(
    model: nn.Sequential,
    criterion: Callable,
    input: Tensor,
    target: Tensor,
    weighting: Weighting[PSDMatrix],
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Currently only works for IWRM supervised training of sequential models.
    """
    model_output, losses, gramian = get_output_loss_and_gramian_supervised_iwrm_sequential(
        model, criterion, input, target
    )
    weights = weighting(gramian)
    weighted_loss = losses @ weights
    weighted_loss.backward()

    return model_output, losses, weights


def get_output_loss_and_gramian_supervised_iwrm_sequential(
    model: nn.Sequential, criterion: Callable, input: Tensor, target: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    bs = input.shape[0]
    outputs = _compute_outputs(criterion, input, model, target)
    grad = torch.ones_like(outputs[-1])
    gramian = torch.zeros(bs, bs, device=grad.device)
    for i, (input, output, layer) in list(
        enumerate(zip(outputs[:-1], outputs[1:], list(model) + [criterion]))
    )[::-1]:
        params = list(layer.parameters())
        if len(params) > 0:

            def get_vjp(input_j, grad_output_j) -> tuple[Tensor, ...]:
                # Note: we use unsqueeze(0) to turn a single activation (or grad_output) into a
                # "batch" of 1 activation (or grad_output). This is because some layers (e.g.
                # nn.Flatten) do not work equivalently if they're provided with a batch or with an
                # element of a batch. We thus always provide them with batches, just of a different
                # size.
                return _vjp_from_module(layer, input_j.unsqueeze(0))(grad_output_j.unsqueeze(0))

            jacobians = torch.vmap(get_vjp)(input, grad)

            assert len(jacobians) == 1

            for jacobian in jacobians[0].values():
                J = jacobian.reshape((bs, -1))
                gramian.addmm_(J, J.T)

        if i == 0:
            break  # Don't try to differentiate with respect to the model's input
        grad = torch.autograd.grad(output, input, grad, retain_graph=True)[0]

    model_output = outputs[-2]
    losses = outputs[-1]
    return model_output, losses, gramian


def _compute_outputs(criterion, input, model: nn.Sequential, target) -> list[Tensor]:
    activations = [input]
    for layer in model:
        activation = layer(activations[-1])
        activations.append(activation)

    losses = criterion(activations[-1], target)
    return activations + [losses]


def _vjp_from_module(module: nn.Module, *inputs) -> Callable:
    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers())}
        return torch.func.functional_call(module, all_state, tuple(inputs))

    return torch.func.vjp(functional_model_call, dict(module.named_parameters()))[1]
