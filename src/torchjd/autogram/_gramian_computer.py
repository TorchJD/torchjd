from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils._pytree import PyTree

from torchjd.autogram._jacobian_computer import JacobianComputer


class GramianComputer(ABC):
    @abstractmethod
    def __call__(
        self,
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> Tensor:
        """Compute what we can for a module and optionally return the gramian if it's ready."""


class JacobianBasedGramianComputer(GramianComputer, ABC):
    def __init__(self, jacobian_computer: JacobianComputer):
        self.jacobian_computer = jacobian_computer

    @staticmethod
    def _to_gramian(jacobian: Tensor) -> Tensor:
        return jacobian @ jacobian.T


class JacobianBasedGramianComputerWithoutCrossTerms(JacobianBasedGramianComputer):
    """
    Stateful JacobianBasedGramianComputer that directly returning the gramian without considering
    cross-terms (except intra-module cross-terms).
    """

    def __call__(
        self,
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> Tensor:
        """Compute what we can for a module and optionally return the gramian if it's ready."""

        jacobian_matrix = self.jacobian_computer(rg_outputs, grad_outputs, args, kwargs)
        return self._to_gramian(jacobian_matrix)
