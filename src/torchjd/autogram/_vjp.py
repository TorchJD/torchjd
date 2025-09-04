from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils._pytree import PyTree, tree_flatten, tree_map_only, tree_unflatten

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


class VJP(ABC):
    """
    Represents an abstract VJP function for a module's forward pass with respect to its parameters.

    :params module: The module to differentiate.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.named_parameters = dict(module.named_parameters(recurse=False))

    @abstractmethod
    def __call__(self, grad_outputs: PyTree, inputs: PyTree) -> dict[str, Tensor]:
        """
        VJP function that takes cotangents and inputs and returns dictionary of names of
        parameters (as given by `module.named_parameters.keys()`) to gradients of the parameters
        for the given cotangents at the given inputs.
        """


class FunctionalVJP(VJP):
    """
    Represents a VJP function for a module's forward pass with respect to its parameters using the
    func api. The __call__ function takes both the inputs and the cotangents that can be vmaped
    jointly in both terms to avoid providing to block diagonal jacobians. The disadvantage of using
    this method is that it computes the forward phase.

    :params module: The module to differentiate.
    """

    def __init__(self, module: nn.Module):
        super().__init__(module)

    def __call__(self, grad_outputs_j: PyTree, inputs_j: PyTree) -> dict[str, Tensor]:
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
        return self._vjp_from_module(inputs_j)(grad_outputs_j)[0]

    def _vjp_from_module(self, inputs: PyTree) -> Callable[[PyTree], tuple[dict[str, Tensor]]]:
        """
        Create a VJP function for a module's forward pass with respect to its parameters.

        Returns a function that computes vector-Jacobian products for the module's parameters given
        fixed inputs. Only parameters with requires_grad=True are included in the differentiation.

        :param inputs: Fixed inputs to the module for the VJP computation.
        :returns: VJP function that takes cotangents and returns parameter gradients.
        """
        requires_grad_named_params = {
            k: v for k, v in self.named_parameters.items() if v.requires_grad
        }
        no_requires_grad_named_params = {
            k: v for k, v in self.named_parameters.items() if not v.requires_grad
        }

        def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
            all_state = {
                **primals,
                **dict(self.module.named_buffers()),
                **no_requires_grad_named_params,
            }
            return torch.func.functional_call(self.module, all_state, inputs)

        return torch.func.vjp(functional_model_call, requires_grad_named_params)[1]


class AutogradVJP(VJP):
    """
    Represents a VJP function for a module's forward pass with respect to its parameters using the
    autograd engine. The __call__ function takes both the inputs and the cotangents but ignores the
    inputs. The main advantage of using this method is that it doesn't require computing the forward
    phase.
    """

    def __init__(self, module: nn.Module, outputs: Sequence[Tensor]):
        super().__init__(module)
        self.outputs = outputs
        self.parameters, self.param_spec = tree_flatten(self.named_parameters)

    def __call__(self, grad_outputs: PyTree, _: PyTree) -> dict[str, Tensor]:
        grads = torch.autograd.grad(
            self.outputs, self.parameters, tree_flatten(grad_outputs)[0], retain_graph=True
        )
        return tree_unflatten(grads, self.param_spec)
