from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils._pytree import PyTree, tree_map_only

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


def get_instance_wise_vjp(module: nn.Module) -> Callable[[PyTree, PyTree], dict[str, Tensor]]:
    """
    Create a VJP function for a module's forward pass with respect to its parameters. The returned
    function takes both the input and the cotangents that can be vmaped jointly in both terms to
    avoid providing to block diagonal jacobians.

    :params module: The module to differentiate.
    :returns: VJP function that takes cotangents and inputs and returns dictionary of names of
        parameters (as given by `module.named_parameters.keys()`) to gradients of the parameters
        for the given cotangents at the given inputs.
    """

    def get_vjp(grad_outputs_j: PyTree, inputs_j: PyTree) -> dict[str, Tensor]:
        # Note: we use unsqueeze(0) to turn a single activation (or grad_output) into a
        # "batch" of 1 activation (or grad_output). This is because some layers (e.g.
        # nn.Flatten) do not work equivalently if they're provided with a batch or with
        # an element of a batch. We thus always provide them with batches, just of a
        # different size.
        inputs_j = tree_map_only(torch.Tensor, lambda x: x.unsqueeze(0), inputs_j)
        grad_outputs_j = tree_map_only(torch.Tensor, lambda x: x.unsqueeze(0), grad_outputs_j)

        # _vjp_from_module returns a function that computes the vjp w.r.t. to the
        # primals (tuple), here the functional has a single primal which is
        # dict(module.named_parameters()). We therefore take the 0'th element to obtain
        # the dict of gradients w.r.t. the module's named_parameters.
        return _vjp_from_module(module, inputs_j)(grad_outputs_j)[0]

    return get_vjp


def _vjp_from_module(
    module: nn.Module, inputs: PyTree
) -> Callable[[PyTree], tuple[dict[str, Tensor]]]:
    """
    Create a VJP function for a module's forward pass with respect to its parameters.

    Returns a function that computes vector-Jacobian products for the module's parameters given
    fixed inputs. Only parameters with requires_grad=True are included in the differentiation.

    :param module: The module to differentiate.
    :param inputs: Fixed inputs to the module for the VJP computation.
    :returns: VJP function that takes cotangents and returns parameter gradients.
    """
    named_params = dict(module.named_parameters(recurse=False))
    requires_grad_named_params = {k: v for k, v in named_params.items() if v.requires_grad}
    no_requires_grad_named_params = {k: v for k, v in named_params.items() if not v.requires_grad}

    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers()), **no_requires_grad_named_params}
        return torch.func.functional_call(module, all_state, inputs)

    return torch.func.vjp(functional_model_call, requires_grad_named_params)[1]
