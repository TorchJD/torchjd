from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils._pytree import PyTree, tree_flatten, tree_map, tree_map_only

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
        self.frozen_params = dict[str, Parameter]()

        for name, param in module.named_parameters(recurse=True):
            if param.requires_grad:
                self.rg_params[name] = param
            else:
                self.frozen_params[name] = param

    def __call__(
        self,
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
    ) -> Tensor:
        return ComputeModuleJacobians.apply(
            self._compute_jacobian, args, kwargs, rg_outputs, *grad_outputs
        )

    @abstractmethod
    def _compute_jacobian(
        self,
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
    ) -> Tensor:
        """
        Computes and returns the Jacobian. The output must be a matrix (2D Tensor).
        """


class FunctionalJacobianComputer(JacobianComputer):
    """
    Represents a function that computes Jacobians for a module's forward pass with respect to its
    parameters using the functional differentiation API. This requires to use vmap, so it's not
    compatible with every module, and it requires to have an extra forward pass to create the vjp
    function.
    """

    def _compute_jacobian(
        self,
        grad_outputs: tuple[Tensor, ...],
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        _: Sequence[Tensor],
    ) -> Tensor:
        grad_outputs_in_dims = (0,) * len(grad_outputs)
        args_in_dims = tree_map(lambda t: 0 if isinstance(t, Tensor) else None, args)
        kwargs_in_dims = tree_map(lambda t: 0 if isinstance(t, Tensor) else None, kwargs)
        in_dims = (grad_outputs_in_dims, args_in_dims, kwargs_in_dims)
        vmapped_vjp = torch.vmap(self._call_on_one_instance, in_dims=in_dims)

        return vmapped_vjp(grad_outputs, args, kwargs)

    def _call_on_one_instance(
        self,
        grad_outputs_j: tuple[Tensor, ...],
        args_j: tuple[PyTree, ...],
        kwargs_j: dict[str, PyTree],
    ) -> Tensor:
        # Note: we use unsqueeze(0) to turn a single activation (or grad_output) into a
        # "batch" of 1 activation (or grad_output). This is because some layers (e.g.
        # nn.Flatten) do not work equivalently if they're provided with a batch or with
        # an element of a batch. We thus always provide them with batches, just of a
        # different size.
        args_j = tree_map_only(torch.Tensor, lambda x: x.unsqueeze(0), args_j)
        kwargs_j = tree_map_only(torch.Tensor, lambda x: x.unsqueeze(0), kwargs_j)
        grad_outputs_j_ = [x.unsqueeze(0) for x in grad_outputs_j]

        def functional_model_call(rg_params: dict[str, Parameter]) -> list[Tensor]:
            all_state = [
                cast(dict[str, Tensor], rg_params),
                dict(self.module.named_buffers()),
                cast(dict[str, Tensor], self.frozen_params),
            ]
            output = torch.func.functional_call(self.module, all_state, args_j, kwargs_j)
            flat_outputs = tree_flatten(output)[0]
            rg_outputs = [t for t in flat_outputs if isinstance(t, Tensor) and t.requires_grad]
            return rg_outputs

        vjp_func = torch.func.vjp(functional_model_call, self.rg_params)[1]

        # vjp_func is a function that computes the vjp w.r.t. to the primals (tuple). Here the
        # functional has a single primal which is dict(module.named_parameters()). We therefore take
        # the 0'th element to obtain the dict of gradients w.r.t. the module's named_parameters.
        gradients = vjp_func(grad_outputs_j_)[0]
        gradient = torch.cat([t.reshape(-1) for t in gradients.values()])
        return gradient


class AutogradJacobianComputer(JacobianComputer):
    """
    Represents a function that computes Jacobians for a module's forward pass with respect to its
    parameters using the autograd engine. The __call__ function takes both the inputs and the
    cotangents but ignores the inputs. The main advantage of using this method is that it doesn't
    require making an extra forward pass.
    """

    def _compute_jacobian(
        self,
        grad_outputs: tuple[Tensor, ...],
        _: tuple[PyTree, ...],
        __: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
    ) -> Tensor:
        flat_rg_params, _ = tree_flatten(self.rg_params)
        grads = torch.autograd.grad(
            rg_outputs,
            flat_rg_params,
            grad_outputs,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        flattened_grads = torch.cat([g.reshape(-1) for g in grads])
        jacobian = flattened_grads.unsqueeze(0)
        return jacobian


class ComputeModuleJacobians(torch.autograd.Function):
    @staticmethod
    def forward(
        compute_jacobian_fn: Callable,
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
        *grad_outputs: Tensor,
    ) -> Tensor:
        # There is no non-batched dimension
        jacobian = compute_jacobian_fn(grad_outputs, args, kwargs, rg_outputs)
        return jacobian

    @staticmethod
    def vmap(
        _,
        in_dims: tuple,
        # tuple[None, tuple[PyTree, ...], dict[str, PyTree], Sequence[int], *tuple[int | None, ...]]
        compute_jacobian_fn: Callable,
        args: tuple[PyTree, ...],
        kwargs: dict[str, PyTree],
        rg_outputs: Sequence[Tensor],
        *jac_outputs: Tensor,
    ) -> tuple[Tensor, None]:
        # There is a non-batched dimension
        # We do not vmap over the args for the non-batched dimension
        in_dims = (in_dims[4:], None, None, None)
        generalized_jacobian = torch.vmap(compute_jacobian_fn, in_dims=in_dims)(
            jac_outputs, args, kwargs, rg_outputs
        )
        shape = generalized_jacobian.shape
        jacobian = generalized_jacobian.reshape([shape[0] * shape[1], -1])
        return jacobian, None

    @staticmethod
    def setup_context(*_) -> None:
        pass
