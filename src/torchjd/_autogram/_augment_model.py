from typing import cast

import torch
from torch import Tensor, nn
from torch.autograd.graph import GradientEdge, get_gradient_edge
from torch.utils._pytree import PyTree, TreeSpec, tree_flatten, tree_unflatten

from torchjd._autogram._gramian_accumulator import GramianAccumulator
from torchjd._autogram._handle import AutogramHandleManager, HandleManager
from torchjd._autogram._target_registry import TargetRegistry
from torchjd._autogram._vjp import get_instance_wise_vjp
from torchjd.aggregation._weighting_bases import PSDMatrix, Weighting

# Note about import from protected _pytree module:
# PyTorch maintainers plan to make pytree public (see
# https://github.com/pytorch/pytorch/issues/65761, https://github.com/pytorch/pytorch/pull/137400).
# It should also come with better speed, because the current implementation is slow, according to
# https://github.com/pytorch/pytorch/issues/65761#issue-1010116111.
# When pytree becomes public, this import will have to be changed with a conditional import (to
# still support older versions of PyTorch where pytree is protected).


def augment_model_for_gramian_based_iwrm(
    model: nn.Module,
    weighting: Weighting[PSDMatrix],
) -> HandleManager:
    """
    Adds module hooks to a model and its child modules so that the backward pass is replaced by a
    step of Gramian-based Jacobian descent automatically.

    After the model has been augmented, the output obtained from it will have an extended
    computation graph that is able to:

    - Compute efficiently the Gramian of the Jacobian of the per-sample losses with respect to the
      model parameters.
    - Extract weights from this Gramian using the provided ``weighting``.
    - Backpropagate these weights for a normal backward pass.

    :param model: The model to augment.
    :param weighting: The object responsible for extracting weights from the Gramian.

    .. admonition::
        Example

        Train a model using Gramian-based Jacobian descent.

            >>> import torch
            >>> from torch.nn import Linear, MSELoss, ReLU, Sequential
            >>> from torch.optim import SGD
            >>>
            >>> from torchjd import augment_model_for_gramian_based_iwrm
            >>> from torchjd.aggregation import UPGrad
            >>>
            >>> # Generate data (8 batches of 16 examples of dim 5) for the sake of the example
            >>> inputs = torch.randn(8, 16, 5)
            >>> targets = torch.randn(8, 16)
            >>>
            >>> model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
            >>> optimizer = SGD(model.parameters())
            >>>
            >>> criterion = MSELoss(reduction="none")
            >>> # TODO: improve this by making weightings public
            >>> weighting = UPGrad().weighting.weighting
            >>> augment_model_for_gramian_based_iwrm(model, weighting)
            >>>
            >>> for input, target in zip(inputs, targets):
            >>>     output = model(input)
            >>>     losses = criterion(output, target)
            >>>
            >>>     optimizer.zero_grad()
            >>>     losses.backward(torch.ones_like(losses))
            >>>     optimizer.step()

        Each call to ``losses.backward(torch.ones_like(losses))`` has computed the Gramian of the
        Jacobian of the losses with respect to the model's parameters, has extracted weights from it
        and has backpropagated these weights to obtain the gradients to use to update the model
        parameters, stored in their ``.grad`` fields. The ``optimizer.step()`` call then updates the
        model parameters based on those ``.grad`` fields.

    .. note::
        If you want to remove the hooks added by ``augment_model_for_gramian_based_iwrm``, you can
        call ``remove()`` on the :class:`~torchjd._autogram._handle.HandleManager` that it returns.

            >>> # Augment the model
            >>> handle = augment_model_for_gramian_based_iwrm(model, weighting)
            >>>
            >>>  # Use it
            >>>  # ...
            >>>
            >>> # De-augment the model
            >>> handle.remove()
            >>> # All hooks added by augment_model_for_gramian_based_iwrm should have been removed
    """
    model_augmenter = _ModelAugmenter(model, weighting)
    model_augmenter.augment()

    return model_augmenter.handle_manager


class HookActivator:
    def __init__(self):
        self.state = True

    def activate(self) -> None:
        self.state = True

    def deactivate(self) -> None:
        self.state = False


class _ModelAugmenter:
    def __init__(self, model: nn.Module, weighting: Weighting[PSDMatrix]):
        self._model = model
        self._weighting = weighting
        self.handle_manager = AutogramHandleManager()

        self._gramian_accumulator = GramianAccumulator()
        self._hook_activator = HookActivator()
        self._target_edges_registry = TargetRegistry()

    def augment(self):
        self._hook_submodules()
        self._hook_model()

    def _hook_submodules(self) -> None:
        for module in self._model.modules():
            if next(module.parameters(recurse=False), None) is None:
                # Skip un-parameterized modules
                continue
            self._hook_module(module)

    def _hook_module(self, module: nn.Module) -> None:
        def module_hook(_, args: PyTree, output: PyTree) -> PyTree:
            if not self._hook_activator.state:
                return output
            flat_outputs, tree_spec = tree_flatten(output)

            if output is None:
                # This can happen only if a module returns no Tensor, for instance some niche usage
                # such as a module that prints something.
                return output

            jacobian_accumulator = _make_jacobian_accumulator(
                module, self._gramian_accumulator, args, tree_spec
            )

            requires_grad_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
            self._gramian_accumulator.track_parameter_paths(requires_grad_params)

            # We only care about running the JacobianAccumulator node, so we need one of its child
            # edges (the edges of the original ouputs of the model) as target. For memory
            # efficiency, we select the smallest one.
            numels = torch.tensor([t.numel() for t in flat_outputs])
            index = cast(int, numels.argmin().item())
            self._target_edges_registry.register(flat_outputs[index])

            return tree_unflatten(jacobian_accumulator.apply(*flat_outputs), tree_spec)

        self.handle_manager.add_handle(module.register_forward_hook(module_hook))

    def _hook_model(self) -> None:

        def model_hook(_, args: PyTree, output: PyTree) -> PyTree:
            if not self._hook_activator.state:
                return output

            input_tensors = [a for a in tree_flatten(args)[0] if isinstance(a, Tensor)]
            excluded_edges = {get_gradient_edge(t) for t in input_tensors if t.requires_grad}
            leaf_targets = self._target_edges_registry.get_leaf_target_edges(excluded_edges)
            flat_outputs, tree_spec = tree_flatten(output)
            autogram_activator = self._make_autogram_activator(flat_outputs, leaf_targets)
            self._deactivate_module_hooks()
            activator_flat_outputs = autogram_activator.apply(*flat_outputs)
            return tree_unflatten(activator_flat_outputs, tree_spec)

        self.handle_manager.add_handle(self._model.register_forward_hook(model_hook))

    def _make_autogram_activator(
        self, flat_outputs: PyTree, leaf_targets: list[GradientEdge]
    ) -> type[torch.autograd.Function]:

        class AutogramActivator(torch.autograd.Function):
            @staticmethod
            def forward(*xs: Tensor) -> tuple[Tensor, ...]:
                return tuple([x.detach() for x in xs])

            @staticmethod
            def setup_context(*_):
                pass

            @staticmethod
            def backward(ctx, *grad_outputs: Tensor):
                _ = torch.autograd.grad(
                    outputs=flat_outputs,
                    inputs=leaf_targets,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                )

                gramian = self._gramian_accumulator.gramian

                # Should never happen, these asserts are temporary for development safety reason.
                assert len(self._gramian_accumulator._path_counter) == 0
                assert len(self._gramian_accumulator._summed_jacobians) == 0
                assert gramian is not None

                # Reset everything that has a state
                self._gramian_accumulator.reset()
                self._hook_activator.activate()
                self._target_edges_registry.reset()

                weights = self._weighting(gramian).unsqueeze(1)
                scaled_grad_outputs = tuple([weights * grad_output for grad_output in grad_outputs])
                return scaled_grad_outputs

        return AutogramActivator

    def _deactivate_module_hooks(self) -> None:
        self._hook_activator.deactivate()


def _make_jacobian_accumulator(
    module: nn.Module,
    gramian_accumulator: GramianAccumulator,
    args: PyTree,
    tree_spec: TreeSpec,
) -> type[torch.autograd.Function]:

    class JacobianAccumulator(torch.autograd.Function):

        activated = True

        @staticmethod
        def forward(*xs: Tensor) -> tuple[Tensor, ...]:
            return tuple([x.detach() for x in xs])

        @staticmethod
        def setup_context(*_):
            pass

        @staticmethod
        def backward(ctx, *flat_grad_outputs: Tensor):
            if not JacobianAccumulator.activated:
                JacobianAccumulator.activated = True
                return flat_grad_outputs

            JacobianAccumulator.activated = False

            grad_outputs = tree_unflatten(flat_grad_outputs, tree_spec)
            jacobians = torch.vmap(get_instance_wise_vjp(module))(grad_outputs, args)

            gramian_accumulator.accumulate_path_jacobians(
                {
                    module.get_parameter(param_name): jacobian
                    for param_name, jacobian in jacobians.items()
                }
            )

            return flat_grad_outputs

    return JacobianAccumulator
