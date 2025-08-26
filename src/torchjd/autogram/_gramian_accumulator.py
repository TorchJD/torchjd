from collections import Counter
from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor


class GramianAccumulator:
    """
    Efficiently accumulates the Gramian of the Jacobian during reverse-mode differentiation.

    Jacobians from multiple graph paths to the same parameter are first summed to obtain the full
    Jacobian w.r.t. a parameter, then its Gramian is computed and accumulated, over parameters, into
    the total Gramian matrix. Intermediate matrices are discarded immediately to save memory.
    """

    def __init__(self) -> None:
        self._gramian: Optional[Tensor] = None
        self._summed_jacobians = dict[Tensor, Tensor]()
        self._path_counter = Counter[Tensor]()

    def reset(self) -> None:
        self._gramian = None
        self._summed_jacobians = {}
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
        gramian = _compute_gramian_of_jacobian(parameter, self._summed_jacobians[parameter])
        if self._gramian is None:
            self._gramian = gramian
        else:
            self._gramian += gramian

    @property
    def gramian(self) -> Optional[Tensor]:
        """
        Get the Gramian matrix accumulated so far.

        :returns: Accumulated Gramian matrix of shape (batch_size, batch_size) or None if nothing
            was accumulated yet.
        """

        return self._gramian


def _compute_gramian_of_jacobian(tensor: Tensor, jacobian: Tensor) -> Tensor:
    permuted_jacobian = jacobian.permute(tuple(reversed(range(jacobian.ndim))))
    indices1 = list(range(jacobian.ndim - tensor.ndim, jacobian.ndim))
    indices2 = list(range(tensor.ndim))[::-1]
    return torch.tensordot(jacobian, permuted_jacobian, dims=(indices1, indices2))
