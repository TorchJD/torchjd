from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch
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

    def compute_jacobian(
        self,
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
    ) -> Tensor:
        return ComputeModuleJacobians.apply(
            self.jacobian_computer, args, kwargs, rg_outputs, *grad_outputs
        )

    @staticmethod
    def _to_gramian(jacobian: Tensor) -> Tensor:
        return jacobian @ jacobian.T


class ComputeModuleJacobians(torch.autograd.Function):
    @staticmethod
    def forward(
        jacobian_computer: JacobianComputer,
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
        *grad_outputs: Tensor,
    ) -> Tensor:
        # There is no non-batched dimension
        jacobian = jacobian_computer(grad_outputs, args, kwargs, rg_outputs)
        return jacobian

    @staticmethod
    def vmap(
        _,
        in_dims: tuple,
        # tuple[None, tuple[PyTree, ...], dict[str, PyTree], Sequence[int], *tuple[int | None, ...]]
        jacobian_computer: JacobianComputer,
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
        *jac_outputs: Tensor,
    ) -> tuple[Tensor, None]:
        # There is a non-batched dimension
        # We do not vmap over the args for the non-batched dimension
        in_dims = (in_dims[4:], None, None, None)
        generalized_jacobian = torch.vmap(jacobian_computer, in_dims=in_dims)(
            jac_outputs, args, kwargs, rg_outputs
        )
        shape = generalized_jacobian.shape
        jacobian = generalized_jacobian.reshape([shape[0] * shape[1], -1])
        return jacobian, None

    @staticmethod
    def setup_context(*_) -> None:
        pass


class JacobianBasedGramianComputerWithoutCrossTerms(JacobianBasedGramianComputer):
    """
    Stateful GramianComputer that waits for all usages to be counted before returning the gramian.
    """

    def __call__(
        self,
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
    ) -> Tensor:
        """Compute what we can for a module and optionally return the gramian if it's ready."""

        return self._to_gramian(self.compute_jacobian(grad_outputs, args, kwargs, rg_outputs))


class JacobianBasedGramianComputerWithCrossTerms(JacobianBasedGramianComputer):
    """
    Stateful GramianComputer that waits for all usages to be counted before returning the gramian.
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

        jacobian_matrix = self.compute_jacobian(grad_outputs, args, kwargs, rg_outputs)

        if self.summed_jacobian is None:
            self.summed_jacobian = jacobian_matrix
        else:
            self.summed_jacobian += jacobian_matrix

        self.remaining_counter -= 1

        if self.remaining_counter == 0:
            gramian = self.summed_jacobian @ self.summed_jacobian.T
            del self.summed_jacobian
            return gramian
        else:
            return None
