from collections.abc import Iterable, Sequence

from torch import Tensor

from ._transform import AccumulateJac, Diagonalize, Init, Jac, OrderedSet
from ._utils import as_checked_ordered_set, check_optional_positive_chunk_size, get_leaf_tensors


def backward(
    tensors: Sequence[Tensor] | Tensor,
    jac_tensors: Sequence[Tensor] | Tensor | None = None,
    inputs: Iterable[Tensor] | None = None,
    retain_graph: bool = False,
    parallel_chunk_size: int | None = None,
) -> None:
    r"""
    Computes the Jacobians of ``tensors`` with respect to ``inputs``, potentially pre-multiplied by
    ``jac_tensors``, and accumulates the results in the ``.jac`` fields of the ``inputs``.

    Mathematically, if ``jac_tensors`` is provided, this function computes the matrix product
    :math:`J_{init} \cdot J`, where :math:`J` is the Jacobian of ``tensors`` w.r.t ``inputs``, and
    :math:`J_{init}` is the concatenation of ``jac_tensors``. If ``jac_tensors`` is ``None``, it
    assumes an Identity matrix, resulting in the full Jacobian.

    :param tensors: The tensor or tensors to differentiate. Should be non-empty.
    :param jac_tensors: The initial Jacobian to backpropagate. If provided, it must have the same
        length and structure as ``tensors``. Each tensor in ``jac_tensors`` must match the shape of
        the corresponding tensor in ``tensors``, with an extra leading dimension representing the
        number of rows of the resulting Jacobian. If ``None``, defaults to the Identity matrix,
        resulting in the standard Jacobian of ``tensors``.
    :param inputs: The tensors with respect to which the Jacobians must be computed. These must have
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
            >>> from torchjd.autojac import backward
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> backward([y1, y2])
            >>>
            >>> param.jac
            tensor([[-1.,  1.],
                    [ 2.,  4.]])

        The ``.jac`` field of ``param`` now contains the Jacobian of
        :math:`\begin{bmatrix}y_1 \\ y_2\end{bmatrix}` with respect to ``param``.

    .. admonition::
        Example

        This is the same example as before, except that we specify the ``jac_tensors`` that correspond
        to the default `None`

            >>> import torch
            >>>
            >>> from torchjd.autojac import backward
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> J1 = torch.tensor([1.0, 0.0])
            >>> J2 = torch.tensor([0.0, 1.0])
            >>>
            >>> backward([y1, y2], jac_tensors=[J1, J2])
            >>>
            >>> param.jac
            tensor([[-1.,  1.],
                    [ 2.,  4.]])

    .. admonition::
        Example

        If ``jac_tensors`` is made of matrices whose first dimension is 1, then this function is
        equivalent to the call ``autograd.grad(y, grad_tensors=weights)`` up to a reshape of the
        output.

            >>> import torch
            >>>
            >>> from torchjd.autojac import backward
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> y = torch.stack([param[0] ** 2, param[1] ** 3])
            >>>
            >>> weights = torch.tensor([[0.5, 1.0]])
            >>> backward(y, jac_tensors=weights)
            >>>
            >>> param.jac
            tensor([[ 1., 12.]])

    .. warning::
        To differentiate in parallel, ``backward`` relies on ``torch.vmap``, which has some
        limitations: `it does not work on the output of compiled functions
        <https://github.com/pytorch/pytorch/issues/138422>`_, `when some tensors have
        <https://github.com/TorchJD/torchjd/issues/184>`_ ``retains_grad=True`` or `when using an
        RNN on CUDA <https://github.com/TorchJD/torchjd/issues/220>`_, for instance. If you
        experience issues with ``backward`` try to use ``parallel_chunk_size=1`` to avoid relying on
        ``torch.vmap``.
    """
    check_optional_positive_chunk_size(parallel_chunk_size)

    tensors_ = as_checked_ordered_set(tensors, "tensors")

    if len(tensors_) == 0:
        raise ValueError("`tensors` cannot be empty")

    if inputs is None:
        inputs_ = get_leaf_tensors(tensors=tensors_, excluded=set())
    else:
        inputs_ = OrderedSet(inputs)

    if jac_tensors is None:
        # Transform that creates gradient outputs containing only ones.
        init = Init(tensors_)
        # Transform that turns the gradients into Jacobians.
        diag = Diagonalize(tensors_)
        jac_tensors_dict = (diag << init)({})
    else:
        jac_tensors_ = as_checked_ordered_set(jac_tensors, "jac_tensors")
        jac_tensors_dict = dict(zip(tensors_, jac_tensors_, strict=True))

    # Transform that computes the required Jacobians.
    jac = Jac(tensors_, inputs_, parallel_chunk_size, retain_graph)
    # Transform that accumulates the result in the .jac field of the inputs.
    accumulate = AccumulateJac()

    (accumulate << jac)(jac_tensors_dict)
