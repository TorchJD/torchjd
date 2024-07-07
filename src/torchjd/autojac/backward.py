from typing import Iterable, Sequence

from torch import Tensor

from torchjd.aggregation import Aggregator

from ._transform import Diagonalize, EmptyTensorDict, Init, Jac, Store, make_aggregation
from ._transform.strategy import UnifyingStrategy
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
    aggregation by ``A`` and stores it in the ``.grad`` fields of the ``inputs``.

    :param tensors: The tensor or tensors to differentiate. Should be non-empty. The Jacobian
        matrices will have one row for each value of each of these tensors.
    :param inputs: The tensors with respect to which the Jacobian must be computed. These must have
        their ``requires_grad`` flag set to ``True``.
    :param A: Aggregator used to reduce the Jacobian into a vector.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``tensors`` will be differentiated in
        parallel at once. If set to `1`, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``.

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
        raise ValueError("`tensors` cannot be an empty iterable of `Tensor`s.")

    _check_retain_graph_compatible_with_chunk_size(tensors, retain_graph, parallel_chunk_size)

    inputs = list(inputs)

    # Transform that creates gradients containing only ones
    init = Init(tensors)

    # Transform that turns the gradients into jacobians
    diag = Diagonalize(tensors)

    # Transform that computes the required jacobians
    jac = Jac(tensors, inputs, parallel_chunk_size, retain_graph)

    # Transform that defines the aggregation of the jacobians into gradients
    aggregate = make_aggregation(UnifyingStrategy(A, inputs))

    # Transform that stores the gradients with respect to the inputs
    store = Store(inputs)

    backward_transform = store << aggregate << jac << diag << init

    backward_transform(EmptyTensorDict())
