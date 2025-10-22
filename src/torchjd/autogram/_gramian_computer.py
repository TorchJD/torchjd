from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from torchjd.autogram._jacobian_computer import JacobianComputer


class GramianComputer(ABC):
    @abstractmethod
    def __call__(
        self,
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
    ) -> Optional[Tensor]:
        """Compute what we can for a module and optionally return the gramian if it's ready."""

    def track_forward_call(self) -> None:
        """Track that the module's forward was called. Necessary in some implementations."""

    def reset(self):
        """Reset state if any. Necessary in some implementations."""


class JacobianBasedGramianComputer(GramianComputer, ABC):
    def __init__(self, jacobian_computer):
        self.jacobian_computer = jacobian_computer

    @staticmethod
    def _to_gramian(matrix: Tensor) -> Tensor:
        """Contracts the last dimension of matrix to make it into a Gramian."""

        indices = list(range(matrix.ndim))
        transposed_matrix = matrix.movedim(indices, indices[::-1])
        return torch.tensordot(matrix, transposed_matrix, dims=([-1], [0]))


class JacobianBasedGramianComputerWithCrossTerms(JacobianBasedGramianComputer):
    """
    Stateful JacobianBasedGramianComputer that waits for all usages to be counted before returning
    the gramian.
    """

    def __init__(self, jacobian_computer: JacobianComputer):
        super().__init__(jacobian_computer)
        self.remaining_counter = 0
        self.summed_jacobian: Optional[Tensor] = None

    def reset(self) -> None:
        self.remaining_counter = 0
        self.summed_jacobian = None

    def track_forward_call(self) -> None:
        self.remaining_counter += 1

    def __call__(
        self, rg_outputs: tuple[Tensor, ...], grad_outputs: tuple[Tensor, ...]
    ) -> Optional[Tensor]:
        """Compute what we can for a module and optionally return the gramian if it's ready."""

        batched_jacobian = self.jacobian_computer(rg_outputs, grad_outputs)
        jacobian = torch.func.debug_unwrap(batched_jacobian, recurse=True)

        if self.summed_jacobian is None:
            self.summed_jacobian = jacobian
        else:
            self.summed_jacobian += jacobian

        self.remaining_counter -= 1

        if self.remaining_counter == 0:
            gramian = self._to_gramian(self.summed_jacobian)
            del self.summed_jacobian
            return gramian
        else:
            return None
