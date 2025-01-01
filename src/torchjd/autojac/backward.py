from typing import Iterable, Sequence

from torch import Tensor

from torchjd.aggregation import Aggregator

from ._transform import Accumulate, Aggregate, Diagonalize, EmptyTensorDict, Init, Jac
from ._utils import _as_tensor_list, _check_optional_positive_chunk_size, _get_leaf_tensors


def backward(
    tensors: Sequence[Tensor] | Tensor,
    aggregator: Aggregator,
    inputs: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    r"""
    Computes the Jacobian of all values in ``tensors`` with respect to all ``inputs``. Computes its
    aggregation by the provided ``aggregator`` and accumulates it in the ``.grad`` fields of the
    ``inputs``.

    :param tensors: The tensor or tensors to differentiate. Should be non-empty. The Jacobian
        matrices will have one row for each value of each of these tensors.
    :param aggregator: Aggregator used to reduce the Jacobian into a vector.
    :param inputs: The tensors with respect to which the Jacobian must be computed. These must have
        their ``requires_grad`` flag set to ``True``. If not provided, defaults to the leaf tensors
        that were used to compute the ``tensors`` parameter.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``tensors`` will be differentiated in
        parallel at once. If set to ``1``, all coordinates will be differentiated sequentially. A
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
            >>> backward([y1, y2], UPGrad())
            >>>
            >>> param.grad
            tensor([0.5000, 2.5000])

        The ``.grad`` field of ``param`` now contains the aggregation of the Jacobian of
        :math:`\begin{bmatrix}y_1 \\ y_2\end{bmatrix}` with respect to ``param``.

    .. warning::
        To differentiate in parallel, ``backward`` relies on ``torch.vmap``, which has some
        limitations: `it does not work on the output of compiled functions
        <https://github.com/pytorch/pytorch/issues/138422>`_, `when some tensors have
        <https://github.com/TorchJD/torchjd/issues/184>`_ ``retains_grad=True`` or `when using an
        RNN on CUDA <https://github.com/TorchJD/torchjd/issues/220>`_, for instance. If you
        experience issues with ``backward`` try to use ``parallel_chunk_size=1`` to avoid relying on
        ``torch.vmap``.
    """
    _check_optional_positive_chunk_size(parallel_chunk_size)

    tensors = _as_tensor_list(tensors)

    if len(tensors) == 0:
        raise ValueError("`tensors` cannot be empty")

    if inputs is None:
        inputs = _get_leaf_tensors(tensors=tensors, excluded=set())
    else:
        inputs = set(inputs)

    # Transform that creates gradient outputs containing only ones.
    init = Init(tensors)

    # Transform that turns the gradients into Jacobians.
    diag = Diagonalize(tensors)

    # Transform that computes the required Jacobians.
    jac = Jac(tensors, inputs, parallel_chunk_size, retain_graph)

    # Transform that aggregates the Jacobians.
    aggregate = Aggregate(aggregator, inputs)

    # Transform that accumulates the result in the .grad field of the inputs.
    accumulate = Accumulate(inputs)

    backward_transform = accumulate << aggregate << jac << diag << init

    backward_transform(EmptyTensorDict())
