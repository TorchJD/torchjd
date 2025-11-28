from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils._pytree import PyTree

from torchjd.autogram._gramian_utils import reshape_gramian
from torchjd.autogram._jacobian_computer import JacobianComputer


class GramianComputer(ABC):
    @abstractmethod
    def __call__(
        self,
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
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
        self.summed_jacobian: Optional[Tensor] = None

    def reset(self) -> None:
        self.remaining_counter = 0
        self.summed_jacobian = None

    def track_forward_call(self) -> None:
        self.remaining_counter += 1

    def __call__(
        self,
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> Optional[Tensor]:
        """Compute what we can for a module and optionally return the gramian if it's ready."""

        jacobian_matrix = self.jacobian_computer(rg_outputs, grad_outputs, args, kwargs)

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


class ModuleBasedGramianComputer(GramianComputer, ABC):
    def __init__(self, module: nn.Module):
        self.module = module

    def __call__(
        self,
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> Tensor:
        gramian = ComputeGramian.apply(
            self._compute_gramian, rg_outputs, grad_outputs, args, kwargs
        )
        return gramian

    @abstractmethod
    def _compute_gramian(
        self,
        rg_outputs: tuple[Tensor, ...],
        jac_outputs1: tuple[Tensor, ...],
        jac_outputs2: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> Tensor:
        """
        If G is the Gramian of the Jacobian of the model's output w.r.t. the parameters, and J1, J2
        are the jac_outputs (Jacobian of losses w.r.t. outputs), then this should compute the matrix
        J1 @ G @ J2.T
        """


class ComputeGramian(torch.autograd.Function):
    @staticmethod
    def forward(
        compute_gramian_fn: Callable[
            [
                tuple[Tensor, ...],
                tuple[Tensor, ...],
                tuple[Tensor, ...],
                tuple[PyTree, ...],
                dict[str, PyTree],
            ],
            Tensor,
        ],
        rg_outputs: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> Tensor:
        # There is no non-batched dimension
        gramian = compute_gramian_fn(rg_outputs, grad_outputs, grad_outputs, args, kwargs)
        return gramian

    @staticmethod
    def vmap(
        _,
        in_dims: tuple[None, None, tuple[int, ...], None, None],
        compute_gramian_fn: Callable,
        rg_outputs: tuple[Tensor, ...],
        jac_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
    ) -> tuple[Tensor, None]:
        # There is a non-batched dimension
        generalized_gramian = torch.vmap(
            torch.vmap(
                compute_gramian_fn,
                in_dims=(None, in_dims[2], None, None, None),
                out_dims=0,
            ),
            in_dims=(None, None, in_dims[2], None, None),
            out_dims=-1,
        )(rg_outputs, jac_outputs, jac_outputs, args, kwargs)
        shape = generalized_gramian.shape
        gramian = reshape_gramian(generalized_gramian, [shape[0] * shape[1]])
        return gramian, None

    @staticmethod
    def setup_context(*_) -> None:
        pass


class LinearBasedGramianComputer(ModuleBasedGramianComputer):
    def __init__(self, module: nn.Linear):
        super().__init__(module)

    def _compute_gramian(
        self,
        _: tuple[Tensor, ...],
        jac_outputs1: tuple[Tensor, ...],
        jac_outputs2: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        __: dict[str, PyTree],
    ) -> Tensor:

        X = args[0]
        dY1 = jac_outputs1[0]
        dY2 = jac_outputs2[0]

        # TODO: add support for ndim==4 or find solution that works for any ndim.
        if dY1.ndim == 2:
            G = torch.einsum(dY1, [0, 2], X, [0, 3], X, [1, 3], dY2, [1, 2], [0, 1])
            if self.module.bias is not None:
                G += torch.einsum(dY1, [0, 2], dY2, [1, 2], [0, 1])
        elif dY1.ndim == 3:  # Typical in transformers
            G = torch.einsum(dY1, [0, 2, 4], X, [0, 2, 5], X, [1, 3, 5], dY2, [1, 3, 4], [0, 1])
            if self.module.bias is not None:
                G += torch.einsum(dY1, [0, 2, 4], dY2, [1, 3, 4], [0, 1])
        else:
            raise ValueError("Higher dimensions not supported. Open an issue if needed.")

        return G
