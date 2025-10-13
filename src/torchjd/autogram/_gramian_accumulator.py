from collections import Counter
from typing import Optional

import torch
from torch import Tensor, nn


class GramianAccumulator:
    """
    Efficiently accumulates the Gramian of the Jacobian during reverse-mode differentiation.

    Jacobians from multiple graph paths to the same parameter are first summed to obtain the full
    Jacobian w.r.t. a parameter, then its Gramian is computed and accumulated, over parameters, into
    the total Gramian matrix. Intermediate matrices are discarded immediately to save memory.
    """

    def __init__(self) -> None:
        self._gramian: Optional[Tensor] = None
        self._summed_jacobians = dict[nn.Module, list[Tensor]]()
        self._path_counter = Counter[nn.Module]()

    def reset(self) -> None:
        self._gramian = None
        self._summed_jacobians = {}
        self._path_counter = Counter()

    def track_module_paths(self, module: nn.Module) -> None:
        """Increment the usage count of the provided module.

        :param module: The module.
        """

        self._path_counter.update([module])

    def accumulate_path_jacobians(self, module: nn.Module, jacobians: list[Tensor]) -> None:
        """
        Add the Jacobians corresponding to all usages of a module.

        :param module: The module.
        :param jacobians: List of Jacobian tensors of a single path.
        """
        if module in self._summed_jacobians:
            self._summed_jacobians[module] = [
                a + b for a, b in zip(self._summed_jacobians[module], jacobians)
            ]
        else:
            self._summed_jacobians[module] = jacobians
        self._path_counter.subtract([module])
        if self._path_counter[module] == 0:
            for jacobian in self._summed_jacobians[module]:
                self._accumulate_one_jacobian_in_gramian(jacobian)
            del self._path_counter[module]
            del self._summed_jacobians[module]

    def _accumulate_one_jacobian_in_gramian(self, jacobian: Tensor) -> None:
        """
        Compute the Gramian of a Jacobian and accumulate it.

        :param jacobian: the Jacobian.
        """
        full_jacobian_matrix = torch.flatten(jacobian, start_dim=1)
        if self._gramian is not None:
            self._gramian.addmm_(full_jacobian_matrix, full_jacobian_matrix.T)
        else:
            self._gramian = torch.mm(full_jacobian_matrix, full_jacobian_matrix.T)

    @property
    def gramian(self) -> Optional[Tensor]:
        """
        Get the Gramian matrix accumulated so far.

        :returns: Accumulated Gramian matrix of shape (batch_size, batch_size) or None if nothing
            was accumulated yet.
        """

        return self._gramian
