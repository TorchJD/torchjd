from torch import nn
from torch.utils.hooks import RemovableHandle as TorchRemovableHandle

from torchjd.aggregation import PSDMatrix, Weighting
from torchjd.autogram._utils.edge_registry import EdgeRegistry
from torchjd.autogram._utils.forward_hooks import ActivableHookFactory, ModelHook, ModuleHook
from torchjd.autogram._utils.gramian_accumulator import GramianAccumulator

from ._handle import RemovableHandle


def augment_model_for_gramian_based_iwrm(
    model: nn.Module,
    weighting: Weighting[PSDMatrix],
) -> RemovableHandle:
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
    :param weighting: The object responsible for extracting weights from the Gramian. You can find
        below a list of available weightings.
    :returns: A :class:`~torchjd.autogram._handle.RemovableHandle` that can be used to return the
        model to its original state.

    .. admonition::
        Example

        Train a model using Gramian-based Jacobian descent.

            >>> import torch
            >>> from torch.nn import Linear, MSELoss, ReLU, Sequential
            >>> from torch.optim import SGD
            >>>
            >>> from torchjd.autogram import augment_model_for_gramian_based_iwrm
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
        parameters, stored in their ``.grad`` fields. The call to ``optimizer.step()`` then updates
        the model parameters based on those ``.grad`` fields.

    .. note::
        The following weightings are supported by autogram:

        * :class:`~torchjd.aggregation.UPGradWeighting`
        * :class:`~torchjd.aggregation.AlignedMTLWeighting`
        * :class:`~torchjd.aggregation.CAGradWeighting`
        * :class:`~torchjd.aggregation.ConstantWeighting`
        * :class:`~torchjd.aggregation.DualProjWeighting`
        * :class:`~torchjd.aggregation.IMTLGWeighting`
        * :class:`~torchjd.aggregation.KrumWeighting`
        * :class:`~torchjd.aggregation.MeanWeighting`
        * :class:`~torchjd.aggregation.MGDAWeighting`
        * :class:`~torchjd.aggregation.PCGradWeighting`
        * :class:`~torchjd.aggregation.RandomWeighting`
        * :class:`~torchjd.aggregation.SumWeighting`
    """

    handles: list[TorchRemovableHandle] = []
    gramian_accumulator = GramianAccumulator()
    activable_hook_factory = ActivableHookFactory()
    target_edges = EdgeRegistry()

    # Add module forward hooks to compute jacobians
    for module in model.modules():
        if next(module.parameters(recurse=False), None) is None:
            # Skip un-parameterized modules
            continue

        module_hook = activable_hook_factory.make_hook_activable(
            ModuleHook(target_edges, gramian_accumulator)
        )
        handle = module.register_forward_hook(module_hook)
        handles.append(handle)

    # Add model forward hook to trigger autogram
    model_hook = activable_hook_factory.make_hook_activable(
        ModelHook(weighting, target_edges, gramian_accumulator, activable_hook_factory)
    )
    handle = model.register_forward_hook(model_hook)
    handles.append(handle)
    handle_manager = RemovableHandle(handles)

    return handle_manager
