from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from torch import Tensor
from torch.utils._pytree import PyTree

from torchjd.autogram._jacobian_computer import JacobianComputer


class GramianComputer(ABC):
    @abstractmethod
    def __call__(
        self,
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
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
    def _to_gramian(jacobian: Tensor) -> Tensor:
        return jacobian @ jacobian.T


class JacobianBasedGramianComputerWithCrossTerms(JacobianBasedGramianComputer):
    """
    Stateful JacobianBasedGramianComputer that waits for all usages to be counted before returning
    the gramian.
    """

    def __init__(self, jacobian_computer: JacobianComputer):
        super().__init__(jacobian_computer)
        self.remaining_counter = 0
        self.summed_jacobian = None

    def reset(self) -> None:
        self.remaining_counter = 0
        self.summed_jacobian = None

    def track_forward_call(self) -> None:
        self.remaining_counter += 1

    def __call__(
        self,
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
    ) -> Optional[Tensor]:
        """Compute what we can for a module and optionally return the gramian if it's ready."""

        jacobian_matrix = self.jacobian_computer(grad_outputs, args, kwargs, rg_outputs)

        if self.summed_jacobian is None:
            self.summed_jacobian = jacobian_matrix
        else:
            self.summed_jacobian += jacobian_matrix

        self.remaining_counter -= 1

        if self.remaining_counter == 0:
            gramian = self._to_gramian(self.summed_jacobian)
            del self.summed_jacobian
            return gramian
        else:
            return None
