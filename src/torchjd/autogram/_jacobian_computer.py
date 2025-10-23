from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils._pytree import tree_flatten

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


class JacobianComputer(ABC):
    """
    Abstract class to computes Jacobians for a module's forward pass with respect to its parameters.

    :params module: The module to differentiate.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.rg_params = dict[str, Parameter]()

        for name, param in module.named_parameters(recurse=True):
            if param.requires_grad:
                self.rg_params[name] = param

    def __call__(self, rg_outputs: tuple[Tensor, ...], grad_outputs: tuple[Tensor, ...]) -> Tensor:
        """Computes and returns the generalized Jacobian, with its parameter dimensions grouped"""

        batched_jacobian = self.compute(rg_outputs, grad_outputs)
        jacobian = torch.func.debug_unwrap(batched_jacobian, recurse=True)
        return jacobian

    @abstractmethod
    def compute(self, rg_outputs: tuple[Tensor, ...], grad_outputs: tuple[Tensor, ...]) -> Tensor:
        """Computes and returns the generalized Jacobian, possibly batched."""


class AutogradJacobianComputer(JacobianComputer):
    """
    JacobianComputer using the autograd engine. The main advantage of using this method is that it
    doesn't require making an extra forward pass.
    """

    def compute(self, rg_outputs: tuple[Tensor, ...], grad_outputs: tuple[Tensor, ...]) -> Tensor:
        flat_rg_params, ___ = tree_flatten(self.rg_params)
        grads = torch.autograd.grad(
            rg_outputs,
            flat_rg_params,
            grad_outputs,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        flattened_grads = torch.cat([g.reshape(-1) for g in grads])
        return flattened_grads
