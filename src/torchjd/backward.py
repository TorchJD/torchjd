from typing import Iterable, Sequence

from torch import Tensor

from ._transform import Diagonalize, EmptyTensorDict, Init, Jac, Store, make_aggregation
from ._transform.strategy import UnifyingStrategy
from .aggregation import Aggregator


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

    :param tensors: The tensor or tensors to differentiate. Should be non-empty. The Jacobians
        matrices will have one row for each value of each of these tensors.
    :param inputs: The tensors with respect to which the Jacobians must be computed. These must have
        their ``requires_grad`` flag set to ``True``.
    :param A: Aggregator to use for the aggregation of the Jacobian.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``tensor`` will be differentiated in
        parallel at once. If set to `1`, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``.
    """
    if not (parallel_chunk_size is None or parallel_chunk_size > 0):
        raise ValueError(
            "`parallel_chunk_size` should be `None` or greater than `0`. (got "
            f"{parallel_chunk_size})"
        )

    if isinstance(tensors, Tensor):
        tensors = [tensors]

    tensors_numel = sum([tensor.numel() for tensor in tensors])
    if parallel_chunk_size is not None and parallel_chunk_size < tensors_numel and not retain_graph:
        raise ValueError(
            "When using `retain_graph=False`, parameter `parallel_chunk_size` must be `None` or "
            "large enough to compute all gradients in parallel."
        )

    if len(tensors) == 0:
        raise ValueError("`tensors` cannot be an empty iterable of `Tensor`s.")

    inputs = list(inputs)

    # Transform that creates gradients containing only ones
    init = Init(tensors)

    # Transform that turns the gradients into jacobians
    diag = Diagonalize(tensors)

    # Transform that computes the required jacobians
    jac = Jac(tensors, inputs, chunk_size=parallel_chunk_size, retain_graph=retain_graph)

    # Transform that defines the aggregation of the jacobians into gradients
    aggregation = make_aggregation(UnifyingStrategy(A, inputs))

    # Transform that stores the gradients with respect to the inputs
    store = Store(inputs)

    backward_transform = store << aggregation << jac << diag << init

    backward_transform(EmptyTensorDict())
