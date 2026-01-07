from collections.abc import Sequence
from typing import Iterable

from torch import Tensor

from torchjd.autojac._transform._base import Transform
from torchjd.autojac._transform._diagonalize import Diagonalize
from torchjd.autojac._transform._init import Init
from torchjd.autojac._transform._jac import Jac
from torchjd.autojac._transform._ordered_set import OrderedSet
from torchjd.autojac._utils import (
    as_checked_ordered_set,
    check_optional_positive_chunk_size,
    get_leaf_tensors,
)


def jac(
    outputs: Sequence[Tensor] | Tensor,
    inputs: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> tuple[Tensor, ...]:
    r"""
    Computes the Jacobian of all values in ``outputs`` with respect to all ``inputs``. Returns the
    result as a tuple, with one element per input tensor.

    :param outputs: The tensor or tensors to differentiate. Should be non-empty. The Jacobian
        matrices will have one row for each value of each of these tensors.
    :param inputs: The tensors with respect to which the Jacobian must be computed. These must have
        their ``requires_grad`` flag set to ``True``. If not provided, defaults to the leaf tensors
        that were used to compute the ``outputs`` parameter.
    :param retain_graph: If ``False``, the graph used to compute the grad will be freed. Defaults to
        ``False``.
    :param parallel_chunk_size: The number of scalars to differentiate simultaneously in the
        backward pass. If set to ``None``, all coordinates of ``outputs`` will be differentiated in
        parallel at once. If set to ``1``, all coordinates will be differentiated sequentially. A
        larger value results in faster differentiation, but also higher memory usage. Defaults to
        ``None``.

    .. admonition::
        Example

        The following example shows how to use ``jac``.

            >>> import torch
            >>>
            >>> from torchjd.autojac import jac
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> jacobians = jac([y1, y2], [param])
            >>>
            >>> jacobians
            (tensor([-1., 1.],
                    [ 2., 4.]]),)

        The returned tuple contains a single tensor (because there is a single param), that is the
        Jacobian of :math:`\begin{bmatrix}y_1 \\ y_2\end{bmatrix}` with respect to ``param``.

    .. warning::
        To differentiate in parallel, ``jac`` relies on ``torch.vmap``, which has some
        limitations: `it does not work on the output of compiled functions
        <https://github.com/pytorch/pytorch/issues/138422>`_, `when some tensors have
        <https://github.com/TorchJD/torchjd/issues/184>`_ ``retains_grad=True`` or `when using an
        RNN on CUDA <https://github.com/TorchJD/torchjd/issues/220>`_, for instance. If you
        experience issues with ``jac`` try to use ``parallel_chunk_size=1`` to avoid relying on
        ``torch.vmap``.
    """

    check_optional_positive_chunk_size(parallel_chunk_size)
    outputs_ = as_checked_ordered_set(outputs, "outputs")

    if inputs is None:
        inputs_ = get_leaf_tensors(tensors=outputs_, excluded=set())
    else:
        inputs_ = OrderedSet(inputs)

    if len(outputs_) == 0:
        raise ValueError("`outputs` cannot be empty")

    if len(inputs_) == 0:
        raise ValueError("`inputs` cannot be empty")

    jac_transform = _create_transform(
        outputs=outputs_,
        inputs=inputs_,
        retain_graph=retain_graph,
        parallel_chunk_size=parallel_chunk_size,
    )

    result = jac_transform({})
    return tuple(val for val in result.values())


def _create_transform(
    outputs: OrderedSet[Tensor],
    inputs: OrderedSet[Tensor],
    retain_graph: bool,
    parallel_chunk_size: int | None,
) -> Transform:
    # Transform that creates gradient outputs containing only ones.
    init = Init(outputs)

    # Transform that turns the gradients into Jacobians.
    diag = Diagonalize(outputs)

    # Transform that computes the required Jacobians.
    jac = Jac(outputs, inputs, parallel_chunk_size, retain_graph)

    return jac << diag << init
