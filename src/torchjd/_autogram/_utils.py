from collections import Counter, deque
from typing import Callable, Iterable

import torch
from torch import Tensor, nn
from torch.autograd.graph import GradientEdge
from torch.nn import Parameter
from torch.utils._pytree import PyTree


class GramianAccumulator:
    """
    Efficiently accumulates the Gramian of the Jacobian during reverse-mode differentiation.

    Jacobians from multiple graph paths to the same parameter are first summed to obtain the full
    Jacobian w.r.t. a parameter, then its Gramian is computed and accumulated, over parameters, into
    the total Gramian matrix. Intermediate matrices are discarded immediately to save memory.
    """

    def __init__(self):
        self._total_gramian = None
        self._full_jacobians = dict()
        self._path_counter = Counter()

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
        if parameter in self._full_jacobians:
            self._full_jacobians[parameter] += jacobian
        else:
            self._full_jacobians[parameter] = jacobian
        self._path_counter.subtract([parameter])
        if self._path_counter[parameter] == 0:
            self._accumulate_gramian(parameter)
            del self._path_counter[parameter]
            del self._full_jacobians[parameter]

    def _accumulate_gramian(self, parameter: Tensor) -> None:
        """
        Compute the Gramian of full Jacobian and accumulate it.

        :param parameter: Parameter whose full Jacobian is available.
        """
        full_jacobian_matrix = torch.flatten(self._full_jacobians[parameter], start_dim=1)
        if self._total_gramian is not None:
            self._total_gramian.addmm_(full_jacobian_matrix, full_jacobian_matrix.T)
        else:
            self._total_gramian = torch.mm(full_jacobian_matrix, full_jacobian_matrix.T)

    @property
    def gramian(self) -> Tensor:
        """
        Get the final accumulated Gramian matrix.

        :returns: Accumulated Gramian matrix of shape (batch_size, batch_size).
        """

        # Should never happen, this assert is temporary for development safety reason.
        assert len(self._path_counter) == 0 and len(self._full_jacobians) == 0
        return self._total_gramian


def next_edges(edge: GradientEdge) -> list[GradientEdge]:
    """
    Get the next gradient edges in the differentiation graph from the given edge.

    :param edge: The current gradient edge.
    """
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]


def targets_to_leaf_targets(
    targets: list[GradientEdge], excluded: set[GradientEdge]
) -> list[GradientEdge]:
    """
    Compute a minimal subset of targets that yields the same differentiation graph traversal: the
    leaf targets. Specifically, this removes targets that are reachable from other targets in the
    differentiation graph, avoiding the need to keep gradients for all targets in memory
    simultaneously.

    :param targets: The target gradient edges for differentiation.
    :param excluded: Gradient edges that stop graph traversal. Modified in-place.
    :returns: Minimal subset of leaf targets.
    """

    targets_ = set(targets)
    nodes_to_traverse = deque(
        (child, target) for target in targets_ for child in next_edges(target)
    )

    already_added = {child for child, _ in nodes_to_traverse}

    while nodes_to_traverse:
        node, origin = nodes_to_traverse.popleft()
        if node in targets_:
            excluded.add(origin)
        else:
            for child in next_edges(node):
                if child not in already_added:
                    nodes_to_traverse.append((child, origin))
                    already_added.add(child)

    return list(targets_ - excluded)


def vjp_from_module(module: nn.Module, inputs: PyTree) -> Callable:
    named_params = dict(module.named_parameters(recurse=False))
    requires_grad_named_params = {k: v for k, v in named_params.items() if v.requires_grad}
    no_requires_grad_named_params = {k: v for k, v in named_params.items() if not v.requires_grad}

    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers()), **no_requires_grad_named_params}
        return torch.func.functional_call(module, all_state, inputs)

    return torch.func.vjp(functional_model_call, requires_grad_named_params)[1]
