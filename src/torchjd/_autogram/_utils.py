from collections import Counter
from typing import Iterable

import torch
from torch import Tensor
from torch.autograd.graph import GradientEdge


class _GramianAccumulator:
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
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]
