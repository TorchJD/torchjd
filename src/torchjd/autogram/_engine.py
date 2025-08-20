from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge

from ._edge_registry import EdgeRegistry
from ._gramian_accumulator import GramianAccumulator
from ._module_hook_manager import ModuleHookManager


class Engine:
    """
    Used for computing the Gramian of the Jacobian of some vector with respect to the direct
    parameters of all provided modules.

    After this object is constructed, the outputs of the provided modules will have an extended
    computation graph that allows to compute efficiently the Gramian of the Jacobian of the
    per-sample losses with respect to the model parameters.

    This Gramian can then be used to extract weights from this Gramian using the provided
    ``weighting`` which in turn can be backpropagated for a normal backward pass. This is the
    reverse Gramian accumulation algorithm.

    :param modules: A collection of modules whose direct (non-recursive) parameters will contribute
        to the Gramian of the Jacobian.

    .. admonition::
        Example

        Train a model using Gramian-based Jacobian descent.

            >>> import torch
            >>> from torch.nn import Linear, MSELoss, ReLU, Sequential
            >>> from torch.optim import SGD
            >>>
            >>> from torchjd.autogram import Engine
            >>> from torchjd.aggregation import UPGradWeighting
            >>>
            >>> # Generate data (8 batches of 16 examples of dim 5) for the sake of the example
            >>> inputs = torch.randn(8, 16, 5)
            >>> targets = torch.randn(8, 16, 1)
            >>>
            >>> model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
            >>> optimizer = SGD(model.parameters())
            >>>
            >>> criterion = MSELoss(reduction="none")
            >>> weighting = UPGradWeighting()
            >>> engine = Engine(model.modules())
            >>>
            >>> for input, target in zip(inputs, targets):
            >>>     output = model(input)
            >>>     losses = criterion(output, target)
            >>>
            >>>     optimizer.zero_grad()
            >>>     gramian = engine.compute_gramian(losses)
            >>>     losses.backward(weighting(gramian))
            >>>     optimizer.step()
    """

    def __init__(
        self,
        modules: Iterable[nn.Module],
    ):
        self._gramian_accumulator = GramianAccumulator()
        self._target_edges = EdgeRegistry()
        self._module_hook_manager = ModuleHookManager(self._target_edges, self._gramian_accumulator)

        self._track_modules(modules)

    def _track_modules(self, modules: Iterable[nn.Module]) -> None:
        _modules = set(modules)

        # Add module forward hooks to compute jacobians
        for module in _modules:
            if next(module.parameters(recurse=False), None) is not None:
                # Skip un-parameterized modules
                self._module_hook_manager.hook_module(module)

    def deaugment_modules(self) -> None:
        """
        Used to de-augment the provided modules. It removes all the hooks that we have added.

        Typical usage is:

        >>> # Augment the model
        >>> engine = Engine(model.modules())
        >>>
        >>>  # Use it
        >>>  # ...
        >>>
        >>> # De-augment the model
        >>> engine.deaugment_modules()
        >>> # All hooks added by augment_model_for_iwrm have now been removed
        """
        self._module_hook_manager.remove_handles()

    def compute_gramian(self, output: Tensor, grad_outputs: Tensor | None = None) -> Tensor:
        """
        Compute the Gramian of the Jacobian of `output` with respect the direct parameters of all
        `modules`.

        :param output: The vector to differentiate. Must have `ndim == 1`.
        :param grad_outputs: The tangents for the differentiation. Default to a vector of 1s of the
            same shape as `output`.
        :returns: the Gramian of the Jacobian of `output` with respect to the direct parameters of
            all `modules`
        """

        if output.ndim != 1:
            raise ValueError(
                "We currently support computing the Gramian with respect to vectors only."
            )

        if grad_outputs is None:
            grad_outputs = torch.ones_like(output)

        self._module_hook_manager.gramian_accumulation_phase = True

        leaf_targets = list(self._target_edges.get_leaf_edges({get_gradient_edge(output)}, set()))

        _ = torch.autograd.grad(
            outputs=output,
            inputs=leaf_targets,
            grad_outputs=grad_outputs,
            retain_graph=True,
        )

        # If the gramian were None, then leaf_targets would be empty, so autograd.grad would
        # have failed. So gramian is necessarily a valid Tensor here.
        gramian = cast(Tensor, self._gramian_accumulator.gramian)

        # Reset everything that has a state
        self._module_hook_manager.gramian_accumulation_phase = False
        self._gramian_accumulator.reset()
        self._target_edges.reset()

        return gramian
