from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import get_gradient_edge

from ._edge_registry import EdgeRegistry
from ._gramian_accumulator import GramianAccumulator
from ._module_hook_manager import ModuleHookManager

_INCOMPATIBLE_MODULE_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
    nn.RNNBase,
    nn.Transformer,
    nn.TransformerEncoder,
    nn.TransformerDecoder,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
)

_TRACK_RUNNING_STATS_MODULE_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LazyInstanceNorm1d,
    nn.LazyInstanceNorm2d,
    nn.LazyInstanceNorm3d,
)


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
            >>> from torchjd.aggregation import UPGradWeighting
            >>> from torchjd.autogram import Engine
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
            >>>     losses = criterion(output, target).squeeze()
            >>>
            >>>     optimizer.zero_grad()
            >>>     gramian = engine.compute_gramian(losses)
            >>>     losses.backward(weighting(gramian))
            >>>     optimizer.step()

    .. warning::
        To use this engine, the modules should respect a few conditions:

        * They should treat the elements of the batch independently. Most common layers respect
          this, but for example `BatchNorm
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_ does not (it
          computes some average and standard deviation over the elements of the batch).
        * Their inputs and outputs can be any PyTree (tensor, tuple or list of tensors, dict of
          tensors, or any nesting of those structures), but each of these tensors must be batched on
          its first dimension. `Transformers
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`_ and `RNNs
          <https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html>`_ are thus not
          supported yet. This is only an implementation issue, so it should be fixed soon (please
          open an issue if you need extra focus on this).
        * They should not perform in-place operations on tensors (for instance you should not use
          ``track_running_stats=True`` in normalization layers).
        * They should not have side-effects during the forward pass (since their forward pass will
          be called twice, the side-effects could be different from what's expected).
        * If they have some randomness during the forward pass, they should not have direct
          trainable parameters. It is, however, perfectly fine for random modules to have child
          modules that have trainable parameters, so if you have a random module with some direct
          parameters, a simple fix is to wrap these parameters into a child module.
        * For maximum efficiency, they should ideally not contain both direct trainable parameters
          and child modules, especially if those direct trainable parameters are used before the
          child modules. You can always wrap those direct trainable parameters into another child
          module to avoid the slow-down.

        If you're building your own architecture, respecting those criterions should be quite easy.
        However, if you're using an existing architecture, you may have to modify it to make it
        compatible with the autogram engine. For instance, you may want to replace `BatchNorm2d
        <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_ layers by
        `GroupNorm <https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html>`_ or
        `InstanceNorm2d
        <https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html>`_ layers.
    """

    def __init__(self, modules: Iterable[nn.Module]):
        self._gramian_accumulator = GramianAccumulator()
        self._target_edges = EdgeRegistry()
        self._module_hook_manager = ModuleHookManager(self._target_edges, self._gramian_accumulator)

        self._hook_modules(modules)

    def _hook_modules(self, modules: Iterable[nn.Module]) -> None:
        _modules = set(modules)

        # Add module forward hooks to compute jacobians
        for module in _modules:
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                self._check_module_is_compatible(module)
                self._module_hook_manager.hook_module(module)

    @staticmethod
    def _check_module_is_compatible(module: nn.Module) -> None:
        if isinstance(module, _INCOMPATIBLE_MODULE_TYPES):
            raise ValueError(
                f"Found a module of type {type(module)}, which is incompatible with the autogram "
                f"engine. The incompatible module types are {_INCOMPATIBLE_MODULE_TYPES} (and their"
                " subclasses)."
            )

        if isinstance(module, _TRACK_RUNNING_STATS_MODULE_TYPES) and module.track_running_stats:
            raise ValueError(
                f"Found a module of type {type(module)}, with `track_running_stats=True`, which is "
                "incompatible with the autogram engine due to performing in-place operations on "
                "tensors and having side-effects during the forward pass. Try setting"
                "`track_running_stats` to `False`."
            )

    def compute_gramian(self, output: Tensor) -> Tensor:
        """
        Compute the Gramian of the Jacobian of `output` with respect the direct parameters of all
        `modules`.

        :param output: The vector to differentiate. Must have `ndim == 1`.
        :returns: the Gramian of the Jacobian of `output` with respect to the direct parameters of
            all `modules`
        """

        reshaped_output = output.reshape([-1])
        return self._compute_square_gramian(reshaped_output)

    def _compute_square_gramian(self, output: Tensor) -> Tensor:
        self._module_hook_manager.gramian_accumulation_phase = True

        leaf_targets = list(self._target_edges.get_leaf_edges({get_gradient_edge(output)}, set()))

        def differentiation(_grad_output: Tensor) -> tuple[Tensor, ...]:
            return torch.autograd.grad(
                outputs=output,
                inputs=leaf_targets,
                grad_outputs=_grad_output,
                retain_graph=True,
            )

        _ = differentiation(torch.ones_like(output))

        # If the gramian were None, then leaf_targets would be empty, so autograd.grad would
        # have failed. So gramian is necessarily a valid Tensor here.
        gramian = cast(Tensor, self._gramian_accumulator.gramian)

        # Reset everything that has a state
        self._module_hook_manager.gramian_accumulation_phase = False
        self._gramian_accumulator.reset()
        self._target_edges.reset()

        return gramian
