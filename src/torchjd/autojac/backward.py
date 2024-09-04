from typing import Iterable, Sequence

from torch import Tensor

from torchjd.aggregation import Aggregator

from ._transform import Accumulate, Aggregate, Diagonalize, EmptyTensorDict, Init, Jac
from ._utils import (
    _as_tensor_list,
    _check_optional_positive_chunk_size,
    _check_retain_graph_compatible_with_chunk_size,
)


def backward(
    tensors: Sequence[Tensor] | Tensor,
    inputs: Iterable[Tensor],
    A: Aggregator,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    r"""
    Computes the Jacobian of all values in ``tensors`` with respect to all ``inputs``. Computes its
    aggregation by ``A`` and accumulates it in the ``.grad`` fields of the ``inputs``.

    :param tensors: The tensor or tensors to differentiate. Should be non-empty. The Jacobian
        matrices will have one row for each value of each of these tensors.
    :param inputs: The tensors with respect to which the Jacobian must be computed. These must have
        their ``requires_grad`` flag set to ``True``.
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

        The following code snippet showcases a simple usage of ``backward``.

            >>> import torch
            >>>
            >>> from torchjd import backward
            >>> from torchjd.aggregation import UPGrad
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> backward([y1, y2], [param], A=UPGrad())
            >>>
            >>> param.grad
            tensor([0.5000, 2.5000])

        The ``.grad`` field of ``param`` now contains the aggregation of the Jacobian of
        :math:`\begin{bmatrix}y_1 \\ y_2\end{bmatrix}` with respect to ``param``.
    """
    _check_optional_positive_chunk_size(parallel_chunk_size)

    tensors = _as_tensor_list(tensors)

    if len(tensors) == 0:
        raise ValueError("`tensors` cannot be empty")

    _check_retain_graph_compatible_with_chunk_size(tensors, retain_graph, parallel_chunk_size)

    inputs = list(inputs)

    # Transform that creates gradient outputs containing only ones.
    init = Init(tensors)

    # Transform that turns the gradients into Jacobians.
    diag = Diagonalize(tensors)

    # Transform that computes the required Jacobians.
    jac = Jac(tensors, inputs, parallel_chunk_size, retain_graph)

    # Transform that aggregates the Jacobians.
    aggregate = Aggregate(A, inputs)

    # Transform that accumulates the result in the .grad field of the inputs.
    accumulate = Accumulate(inputs)

    backward_transform = accumulate << aggregate << jac << diag << init

    backward_transform(EmptyTensorDict())
