from typing import Iterable, Sequence

from torch import Tensor

from torchjd.aggregation import Aggregator
from torchjd.autojac._transform import (
    EmptyTensorDict,
    Grad,
    Gradients,
    Init,
    Jac,
    Stack,
    Store,
    Subset,
    Transform,
    make_aggregation,
)
from torchjd.autojac._transform.strategy import UnifyingStrategy

from ._utils import (
    _as_tensor_list,
    _check_optional_positive_chunk_size,
    _check_retain_graph_compatible_with_chunk_size,
)


def multi_task_backward(
    tasks_losses: Sequence[Tensor],
    shared_parameters: Iterable[Tensor],
    shared_representations: Sequence[Tensor] | Tensor,
    tasks_parameters: Sequence[Iterable[Tensor]],
    A: Aggregator,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    r"""
    Used for Multi-task learning, this computes the gradient of the ``tasks_losses`` with respect to
    the ``tasks_parameters`` and stores them in their ``.grad`` fields. It then computes the
    Jacobian of the ``tasks_losses`` with respect to the ``shared_parameters`` and stores its
    aggregation by ``A`` into their ``.grad`` fields. The ``tasks_losses`` should be computed from
    ``shared_activation`` and ``tasks_parameters``.


    .. admonition::
        Example

        The following code snippet showcases a simple usage of ``multi_task_backward``.

            >>> import torch
            >>>
            >>> from torchjd import multi_task_backward
            >>> from torchjd.aggregation import UPGrad
            >>>
            >>> p0 = torch.tensor([1.0, 2.0], requires_grad=True)
            >>> p1 = torch.tensor([1.0, 2.0], requires_grad=True)
            >>> p2 = torch.tensor([3.0, 4.0], requires_grad=True)
            >>> shared_parameters = [p0]
            >>> tasks_parameters = [[p1], [p2]]
            >>>
            >>> # Compute arbitrary quantities that are function of param
            >>> r1 = torch.tensor([-1.0, 1.0]) @ p0
            >>> r2 = (p0**2).sum() + p0.norm()
            >>> shared_representations = [r1, r2]
            >>>
            >>> l1 = torch.stack((r1 * p1[0], r2 * p1[1]))
            >>> l2 = r1 * p2[0] + r2 * p2[1]
            >>> tasks_losses = [l1, l2]
            >>>
            >>> multi_task_backward(
            ...     tasks_losses=tasks_losses,
            ...     shared_parameters=shared_parameters,
            ...     shared_representations=shared_representations,
            ...     tasks_parameters=tasks_parameters,
            ...     A=UPGrad(),
            ... )
            >>>
            >>> p0.grad, p1.grad, p2.grad
            (tensor([ 5.3416, 16.6833]), tensor([1.0000, 7.2361]), tensor([1.0000, 7.2361]))

        The ``.grad`` field of ``param`` are now populated.

    :param tasks_losses: The losses of each task. Should contain one tensor per task, and
        therefore must be non-empty. The tensors need not be scalars as the backpropagation is
        initialized with 1, it is equivalent to provide their sums.
    :param shared_parameters: The tensors with respect to which the Jacobian must be computed, i.e.
        the Jacobian matrix will have one column per value in these tensors. These must have their
        ``requires_grad`` flag set to ``True``.
    :param shared_representations: The last shared representation of all tasks. Should be non-empty.
    :param tasks_parameters: The tensors with respect to which the gradients must be computed. There
        must be one collection of parameters per task, therefore the length must match with
        ``tasks_losses``. These must have their ``requires_grad`` flag set to ``True``.
    :param A: Aggregator to use for the aggregation of the Jacobian.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``tensors`` will be differentiated in
        parallel at once. If set to `1`, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``.
    """
    _check_optional_positive_chunk_size(parallel_chunk_size)

    shared_representations = _as_tensor_list(shared_representations)

    if len(shared_representations) == 0:
        raise ValueError("`shared_representations` cannot be an empty iterable of `Tensor`s.")

    _check_retain_graph_compatible_with_chunk_size(
        shared_representations, retain_graph, parallel_chunk_size
    )

    if len(tasks_losses) == 0:
        raise ValueError("`tasks_losses` cannot be an empty sequence of `Tensors`s.")
    if len(tasks_losses) != len(tasks_parameters):
        raise ValueError("`tasks_losses` and `tasks_parameters` should have the same size.")

    shared_parameters = list(shared_parameters)
    tasks_parameters = [list(task_input) for task_input in tasks_parameters]

    # Transforms that store gradient of the losses w.r.t. tasks specific parameters into their
    # ``.grad`` fields and backpropagate the gradient of the losses w.r.t. to the shared
    # representations.
    task_transforms = [
        _make_task_transform(
            shared_representations,
            task_parameters,
            tensor,
            retain_graph,
        )
        for task_parameters, tensor in zip(tasks_parameters, tasks_losses)
    ]

    # Transform that stacks the gradients of the losses w.r.t. shared representations into a
    # Jacobian
    stack = Stack(task_transforms)

    # Transform that computes the Jacobians of the shared parameters
    jac = Jac(shared_representations, shared_parameters, parallel_chunk_size, retain_graph)

    # Transform that defines the aggregation of the jacobians into gradients
    aggregate = make_aggregation(UnifyingStrategy(A, shared_parameters))

    # Transform that stores the gradients with respect to the shared parameters
    store = Store(shared_parameters)

    backward_transform = store << aggregate << jac << stack

    backward_transform(EmptyTensorDict())


def _make_task_transform(
    shared_representations: list[Tensor],
    task_parameters: list[Tensor],
    losses: Tensor,
    retain_graph: bool,
) -> Transform[EmptyTensorDict, Gradients]:
    to_differentiate = task_parameters + shared_representations

    # Transform that initializes the gradient output to 1.
    init = Init([losses])

    # Transform that computes the gradients of task losses w.r.t. the task specific parameters and
    # the shared representations
    grad = Grad([losses], to_differentiate, retain_graph)

    # Transform that stores the task specific parameters into their ``.grad`` fields.
    store = Store(task_parameters) << Subset(task_parameters, to_differentiate)

    # Transform that backpropagate the gradients of the losses w.r.t. the shared representations
    backpropagate = Subset(shared_representations, to_differentiate)

    # Transform that stores gradient of the losses w.r.t. task specific parameters into their
    # ``.grad`` fields and backpropagate the gradient of the losses w.r.t. to the shared
    # representations.
    backward_task = (backpropagate | store) << grad << init
    return backward_task
