from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils._pytree import PyTree, tree_flatten, tree_map_only, tree_unflatten

from torchjd.autogram._module_utils import get_used_params

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


class VJP(ABC):
    """Represents an abstract VJP function."""

    @abstractmethod
    def __call__(
        self, grad_outputs: tuple[Tensor, ...], args: tuple[PyTree, ...], kwargs: dict[str, PyTree]
    ) -> dict[str, Tensor]:
        """
        Computes and returns the dictionary of parameter names to their gradients for the given
        grad_outputs (cotangents) and at the given inputs.
        """


class ModuleVJP(VJP, ABC):
    """
    Represents an abstract VJP function for a module's forward pass with respect to its parameters.

    :params module: The module to differentiate.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.trainable_params, self.frozen_params = get_used_params(module)


class FunctionalVJP(ModuleVJP):
    """
    Represents a VJP function for a module's forward pass with respect to its parameters using the
    functional differentiation API. This requires to use vmap, so it's not compatible with
    every module, and it requires to have an extra forward pass to create the vjp function.
    """

    def __init__(self, module: nn.Module, in_dims: tuple[PyTree, ...]):
        super().__init__(module)
        self.vmapped_vjp = torch.vmap(self._call_on_one_instance, in_dims=in_dims)

    def __call__(
        self, grad_outputs: tuple[Tensor, ...], args: tuple[PyTree, ...], kwargs: dict[str, PyTree]
    ) -> dict[str, Tensor]:
        return self.vmapped_vjp(grad_outputs, args, kwargs)

    def _call_on_one_instance(
        self,
        grad_outputs_j: tuple[Tensor, ...],
        args_j: tuple[PyTree, ...],
        kwargs_j: dict[str, PyTree],
    ) -> dict[str, Tensor]:
        # Note: we use unsqueeze(0) to turn a single activation (or grad_output) into a
        # "batch" of 1 activation (or grad_output). This is because some layers (e.g.
        # nn.Flatten) do not work equivalently if they're provided with a batch or with
        # an element of a batch. We thus always provide them with batches, just of a
        # different size.
        args_j = tree_map_only(torch.Tensor, lambda x: x.unsqueeze(0), args_j)
        kwargs_j = tree_map_only(torch.Tensor, lambda x: x.unsqueeze(0), kwargs_j)
        grad_outputs_j_ = [x.unsqueeze(0) for x in grad_outputs_j]

        def functional_model_call(trainable_params: dict[str, Parameter]) -> list[Tensor]:
            all_state = {
                **trainable_params,
                **dict(self.module.named_buffers()),
                **self.frozen_params,
            }
            output = torch.func.functional_call(self.module, all_state, args_j, kwargs_j)
            flat_outputs = tree_flatten(output)[0]
            rg_outputs = [t for t in flat_outputs if isinstance(t, Tensor) and t.requires_grad]
            return rg_outputs

        vjp_func = torch.func.vjp(functional_model_call, self.trainable_params)[1]

        # vjp_func is a function that computes the vjp w.r.t. to the primals (tuple). Here the
        # functional has a single primal which is dict(module.named_parameters()). We therefore take
        # the 0'th element to obtain the dict of gradients w.r.t. the module's named_parameters.
        return vjp_func(grad_outputs_j_)[0]


class AutogradVJP(ModuleVJP):
    """
    Represents a VJP function for a module's forward pass with respect to its parameters using the
    autograd engine. The __call__ function takes both the inputs and the cotangents but ignores the
    inputs. The main advantage of using this method is that it doesn't require making an extra
    forward pass.
    """

    def __init__(self, module: nn.Module, rg_outputs: Sequence[Tensor]):
        super().__init__(module)

        self.rg_outputs = rg_outputs
        self.flat_trainable_params, self.param_spec = tree_flatten(self.trainable_params)

    def __call__(
        self, grad_outputs: tuple[Tensor, ...], _: tuple[PyTree, ...], __: dict[str, PyTree]
    ) -> dict[str, Tensor]:
        grads = torch.autograd.grad(
            self.rg_outputs,
            self.flat_trainable_params,
            grad_outputs,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        return tree_unflatten(grads, self.param_spec)
