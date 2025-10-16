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


class LinearBasedGramianComputer(GramianComputer):
    def __init__(self, module: nn.Linear):
        self.module = module

    def __call__(
        self,
        _: tuple[Tensor, ...],
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        __: dict[str, PyTree],
    ) -> Optional[Tensor]:

        X = args[0]
        dY = grad_outputs[0]

        gramian = ComputeLinearGramian.apply(self._compute_gramian, dY, X)
        return gramian

    def _compute_gramian(self, dY1: Tensor, dY2: Tensor, X: Tensor) -> Tensor:
        """
        X is a matrix of shape [k, n] and dY1, dY2 are matrices of shape [k, m].
        Returns the dY1 @ G @ dY2 where G is the Gramian of the Jacobian of the module output w.r.t.
        to the module params.
        """

        # TODO: add support for ndim==4 or find solution that works for any ndim.
        if dY1.ndim == 1:
            # TODO: not sure that this even works
            G_b = torch.einsum("k,k->", dY1, dY2)
            G_W = torch.einsum("k,l,l,k->", dY1, X, X, dY2)
        elif dY1.ndim == 2:
            G_b = torch.einsum("ak,ik->ai", dY1, dY2)
            G_W = torch.einsum("ak,al,il,ik->ai", dY1, X, X, dY2)
        elif dY1.ndim == 3:  # Typical in transformers
            G_b = torch.einsum("abk,ijk->ai", dY1, dY2)
            G_W = torch.einsum("abk,abl,ijl,ijk->ai", dY1, X, X, dY2)
        else:
            raise ValueError("Higher dimensions not supported. Open an issue if needed.")

        return G_b + G_W


class ComputeLinearGramian(torch.autograd.Function):
    @staticmethod
    def forward(
        compute_gramian_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        dY: Tensor,
        X: Tensor,
    ) -> Tensor:
        # There is no non-batched dimension
        gramian = compute_gramian_fn(dY, dY, X)
        return gramian

    @staticmethod
    def vmap(
        _,
        in_dims: tuple[None, tuple[int, ...], None],
        compute_gramian_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        dY: Tensor,
        X: Tensor,
    ) -> tuple[Tensor, None]:
        # There is a non-batched dimension
        generalized_gramian = torch.vmap(
            torch.vmap(
                compute_gramian_fn,
                in_dims=(in_dims[1], None, None),
                out_dims=0,
            ),
            in_dims=(None, in_dims[1], None),
            out_dims=-1,
        )(dY, dY, X)
        shape = dY.shape
        gramian = reshape_gramian(generalized_gramian, [shape[0] * shape[1]])
        return gramian, None

    @staticmethod
    def setup_context(*_) -> None:
        pass
