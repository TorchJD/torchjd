from typing import Iterable, Sequence

from torch import Tensor

from torchjd.aggregation import Aggregator

from ._transform import (
    Accumulate,
    Aggregate,
    EmptyTensorDict,
    Grad,
    Gradients,
    Init,
    Jac,
    Select,
    Stack,
    Transform,
)
from ._utils import (
    _as_tensor_list,
    _check_optional_positive_chunk_size,
    _check_retain_graph_compatible_with_chunk_size,
)


def mtl_backward(
    losses: Sequence[Tensor],
    features: Sequence[Tensor] | Tensor,
    tasks_params: Sequence[Iterable[Tensor]],
    shared_params: Iterable[Tensor],
    A: Aggregator,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    """
    In the context of Multi-Task Learning (MTL), we often have a shared feature extractor followed
    by several task-specific heads. A loss can then be computed for each task.

    This function computes the gradient of each task-specific loss with respect to its task-specific
    parameters and accumulates it in their ``.grad`` fields. Then, it computes the Jacobian of all
    losses with respect to the shared parameters, aggregates it and accumulates the result in their
    ``.grad`` fields.

    :param losses: The task losses. The Jacobian matrix will have one row per loss.
    :param features: The last shared representation used for all tasks, as given by the feature
        extractor. Should be non-empty.
    :param tasks_params: The parameters of each task-specific head. Their ``requires_grad`` flags
        must be set to ``True``.
    :param shared_params: The parameters of the shared feature extractor. The Jacobian matrix will
        have one column for each value in these tensors. Their ``requires_grad`` flags must be set
        to ``True``.
    :param A: Aggregator used to reduce the Jacobian into a vector.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``tensors`` will be differentiated in
        parallel at once. If set to ``1``, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``. If ``parallel_chunk_size`` is not large enough to differentiate all tensors
        simultaneously, ``retain_graph`` has to be set to ``True``.

    .. admonition::
        Example

        A usage example of ``mtl_backward`` is provided in
        :doc:`Multi-Task Learning (MTL) <../../examples/mtl>`.
    """

    _check_optional_positive_chunk_size(parallel_chunk_size)

    features = _as_tensor_list(features)

    if len(features) == 0:
        raise ValueError("`features` cannot be empty.")

    _check_retain_graph_compatible_with_chunk_size(features, retain_graph, parallel_chunk_size)

    _check_losses_are_scalar(losses)

    if len(losses) == 0:
        raise ValueError("`tasks_losses` cannot be empty")
    if len(losses) != len(tasks_params):
        raise ValueError("`tasks_losses` and `tasks_params` should have the same size.")

    shared_params = list(shared_params)
    tasks_params = [list(task_params) for task_params in tasks_params]

    # Task-specific transforms. Each of them computes and accumulates the gradient of the task's
    # loss w.r.t. the task's specific parameters, and computes and backpropagates the gradient of
    # the losses w.r.t. the shared representations.
    task_transforms = [
        _make_task_transform(
            features,
            task_params,
            loss,
            retain_graph,
        )
        for task_params, loss in zip(tasks_params, losses)
    ]

    # Transform that stacks the gradients of the losses w.r.t. the shared representations into a
    # Jacobian.
    stack = Stack(task_transforms)

    # Transform that computes the Jacobians of the losses w.r.t. the shared parameters.
    jac = Jac(features, shared_params, parallel_chunk_size, retain_graph)

    # Transform that aggregates the Jacobians.
    aggregate = Aggregate(A, shared_params)

    # Transform that accumulates the result in the .grad field of the shared parameters.
    accumulate = Accumulate(shared_params)

    backward_transform = accumulate << aggregate << jac << stack

    backward_transform(EmptyTensorDict())


def _make_task_transform(
    features: list[Tensor],
    tasks_params: list[Tensor],
    loss: Tensor,
    retain_graph: bool,
) -> Transform[EmptyTensorDict, Gradients]:
    # Tensors with respect to which we compute the gradients.
    to_differentiate = tasks_params + features

    # Transform that initializes the gradient output to 1.
    init = Init([loss])

    # Transform that computes the gradients of the loss w.r.t. the task-specific parameters and
    # the features.
    grad = Grad([loss], to_differentiate, retain_graph)

    # Transform that accumulates the gradients w.r.t. the task-specific parameters into their
    # .grad fields.
    accumulate = Accumulate(tasks_params) << Select(tasks_params, to_differentiate)

    # Transform that backpropagates the gradients of the losses w.r.t. the features.
    backpropagate = Select(features, to_differentiate)

    # Transform that accumulates the gradient of the losses w.r.t. the task-specific parameters into
    # their .grad fields and backpropagates the gradient of the losses w.r.t. to the features.
    backward_task = (backpropagate | accumulate) << grad << init
    return backward_task


def _check_losses_are_scalar(losses: Sequence[Tensor]) -> None:
    for loss in losses:
        if loss.ndim > 0:
            raise ValueError("`losses` should contain only scalars.")
