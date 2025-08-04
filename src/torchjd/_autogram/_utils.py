from abc import ABC, abstractmethod
from collections import Counter, deque
from typing import Callable, Iterable

import torch
from torch import Tensor, nn
from torch.autograd.graph import GradientEdge, get_gradient_edge
from torch.nn import Parameter
from torch.utils._pytree import PyTree, tree_map
from torch.utils.hooks import RemovableHandle


class GramianAccumulator:
    """
    Efficiently accumulates the Gramian of the Jacobian during reverse-mode differentiation.

    Jacobians from multiple graph paths to the same parameter are first summed to obtain the full
    Jacobian w.r.t. a parameter, then its Gramian is computed and accumulated, over parameters, into
    the total Gramian matrix. Intermediate matrices are discarded immediately to save memory.
    """

    def __init__(self) -> None:
        self._gramian: Tensor | None = None
        self._summed_jacobians = dict[Tensor, Tensor]()
        self._path_counter = Counter[Tensor]()

    def track_parameter_paths(self, parameters: Iterable[Tensor]) -> None:
        """
        Register parameters and count their paths in the computational graph.

        :param parameters: Parameter tensors to track. Duplicates increase path count.
        """
        self._path_counter.update(parameters)

    def accumulate_path_jacobians(self, path_jacobians: dict[Tensor, Tensor]) -> None:
        """
        Add path Jacobians for multiple parameters.

        :param path_jacobians: Dictionary mapping parameters to Jacobian tensors of a single path.
        """
        for parameter, jacobian in path_jacobians.items():
            self._accumulate_path_jacobian(parameter, jacobian)

    def _accumulate_path_jacobian(self, parameter: Tensor, jacobian: Tensor) -> None:
        """
        Add path Jacobian for a parameter. In case the full Jacobian is computed, accumulate its
        Gramian.

        :param parameter: The parameter.
        :param jacobian: path Jacobian with respect to the parameter.
        """
        if parameter in self._summed_jacobians:
            self._summed_jacobians[parameter] += jacobian
        else:
            self._summed_jacobians[parameter] = jacobian
        self._path_counter.subtract([parameter])
        if self._path_counter[parameter] == 0:
            self._accumulate_gramian(parameter)
            del self._path_counter[parameter]
            del self._summed_jacobians[parameter]

    def _accumulate_gramian(self, parameter: Tensor) -> None:
        """
        Compute the Gramian of full Jacobian and accumulate it.

        :param parameter: Parameter whose full Jacobian is available.
        """
        full_jacobian_matrix = torch.flatten(self._summed_jacobians[parameter], start_dim=1)
        if self._gramian is not None:
            self._gramian.addmm_(full_jacobian_matrix, full_jacobian_matrix.T)
        else:
            self._gramian = torch.mm(full_jacobian_matrix, full_jacobian_matrix.T)

    @property
    def gramian(self) -> Tensor | None:
        """
        Get the Gramian matrix accumulated so far.

        :returns: Accumulated Gramian matrix of shape (batch_size, batch_size) or None if nothing
            was accumulated yet.
        """

        return self._gramian


class TargetRegistry:
    """
    Tracks the targets for the second backward phase of the autogram algorithm. Enables computing a
    minimally sufficient subset of leaf targets.
    """

    def __init__(self) -> None:
        self._target_edges: set[GradientEdge] = set()

    def register(self, target: Tensor) -> None:
        """
        Track the GradientEdge of the provided target.

        :param target: Tensor to track.
        """
        self._target_edges.add(get_gradient_edge(target))

    def get_leaf_target_edges(self, excluded: set[GradientEdge]) -> list[GradientEdge]:
        """
        Compute a minimal subset of targets that yields the same differentiation graph traversal:
        the leaf targets. Specifically, this removes targets that are reachable from other targets
        in the differentiation graph, avoiding the need to keep gradients for all targets in memory
        simultaneously.

        :param excluded: Gradient edges that stop graph traversal. Modified in-place.
        :returns: Minimal subset of leaf targets.
        """
        nodes_to_traverse = deque(
            (child, target) for target in self._target_edges for child in next_edges(target)
        )

        already_added = {child for child, _ in nodes_to_traverse}

        while nodes_to_traverse:
            node, origin = nodes_to_traverse.popleft()
            if node in self._target_edges:
                excluded.add(origin)
            else:
                for child in next_edges(node):
                    if child not in already_added:
                        nodes_to_traverse.append((child, origin))
                        already_added.add(child)

        return list(self._target_edges - excluded)


def next_edges(edge: GradientEdge) -> list[GradientEdge]:
    """
    Get the next gradient edges in the differentiation graph from the given edge.

    :param edge: The current gradient edge.
    """
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]


def _vjp_from_module(
    module: nn.Module, inputs: PyTree
) -> Callable[[PyTree], tuple[dict[str, Tensor]]]:
    """
    Create a VJP function for a module's forward pass with respect to its parameters.

    Returns a function that computes vector-Jacobian products for the module's parameters given
    fixed inputs. Only parameters with requires_grad=True are included in the differentiation.

    :param module: The module to differentiate.
    :param inputs: Fixed inputs to the module for the VJP computation.
    :returns: VJP function that takes cotangents and returns parameter gradients.
    """
    named_params = dict(module.named_parameters(recurse=False))
    requires_grad_named_params = {k: v for k, v in named_params.items() if v.requires_grad}
    no_requires_grad_named_params = {k: v for k, v in named_params.items() if not v.requires_grad}

    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers()), **no_requires_grad_named_params}
        return torch.func.functional_call(module, all_state, inputs)

    return torch.func.vjp(functional_model_call, requires_grad_named_params)[1]


def get_instance_wise_vjp(module: nn.Module) -> Callable[[PyTree, PyTree], dict[str, Tensor]]:
    """
    Create a VJP function for a module's forward pass with respect to its parameters. The returned
    function takes both the input and the cotangents that can be vmaped jointly in both terms to
    avoid providing to block diagonal jacobians.

    :params module: The module to differentiate.
    :returns: VJP function that takes cotangents and inputs and returns dictionary of names of
        parameters (as given by `module.named_parameters.keys()`) to gradients of the parameters
        for the given cotangents at the given inputs.
    """

    def get_vjp(grad_outputs_j: PyTree, inputs_j: PyTree) -> dict[str, Tensor]:
        # Note: we use unsqueeze(0) to turn a single activation (or grad_output) into a
        # "batch" of 1 activation (or grad_output). This is because some layers (e.g.
        # nn.Flatten) do not work equivalently if they're provided with a batch or with
        # an element of a batch. We thus always provide them with batches, just of a
        # different size.
        inputs_j = tree_map(lambda x: x.unsqueeze(0), inputs_j)
        grad_outputs_j = tree_map(lambda x: x.unsqueeze(0), grad_outputs_j)

        # _vjp_from_module returns a function that computes the vjp w.r.t. to the
        # primals (tuple), here the functional has a single primal which is
        # dict(module.named_parameters()). We therefore take the 0'th element to obtain
        # the dict of gradients w.r.t. the module's named_parameters.
        return _vjp_from_module(module, inputs_j)(grad_outputs_j)[0]

    return get_vjp


class HandleManager(ABC):
    @abstractmethod
    def remove(self):
        """
        Remove handles from a model. This can be used to de-augment a model.
        """


class AutogramHandleManager(HandleManager):
    """
    Private `HandleManager` that is used to track Module hooks' handles to de-augment a model that
    was augmented for autogram.
    """

    def __init__(self) -> None:
        self._handles: list[RemovableHandle] = []

    def add_handle(self, handle: RemovableHandle) -> None:
        self._handles.append(handle)

    def remove(self):
        for handle in self._handles:
            handle.remove()
