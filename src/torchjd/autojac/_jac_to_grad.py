from collections import deque
from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor, nn

from torchjd._linalg import PSDMatrix, compute_gramian
from torchjd.aggregation import Aggregator
from torchjd.aggregation._aggregator_bases import GramianWeightedAggregator

from ._accumulation import TensorWithJac, accumulate_grads, is_tensor_with_jac


def jac_to_grad(
    tensors: Iterable[Tensor], aggregator: Aggregator, retain_jac: bool = False
) -> None:
    r"""
    Aggregates the Jacobians stored in the ``.jac`` fields of ``tensors`` and accumulates the result
    into their ``.grad`` fields.

    :param tensors: The tensors whose ``.jac`` fields should be aggregated. All Jacobians must
        have the same first dimension (e.g. number of losses).
    :param aggregator: The aggregator used to reduce the Jacobians into gradients.
    :param retain_jac: Whether to preserve the ``.jac`` fields of the tensors after they have been
        used. Defaults to ``False``.

    .. note::
        This function starts by "flattening" the ``.jac`` fields into matrices (i.e. flattening all
        of their dimensions except the first one), then concatenates those matrices into a combined
        Jacobian matrix. The aggregator is then used on this matrix, which returns a combined
        gradient vector, that is split and reshaped to fit into the ``.grad`` fields of the tensors.

    .. admonition::
        Example

        This example shows how to use ``jac_to_grad`` after a call to ``backward``

            >>> import torch
            >>>
            >>> from torchjd.aggregation import UPGrad
            >>> from torchjd.autojac import backward, jac_to_grad
            >>>
            >>> param = torch.tensor([1., 2.], requires_grad=True)
            >>> # Compute arbitrary quantities that are function of param
            >>> y1 = torch.tensor([-1., 1.]) @ param
            >>> y2 = (param ** 2).sum()
            >>>
            >>> backward([y1, y2])  # param now has a .jac field
            >>> jac_to_grad([param], aggregator=UPGrad())  # param now has a .grad field
            >>> param.grad
            tensor([-1.,  1.])

        The ``.grad`` field of ``param`` now contains the aggregation (by UPGrad) of the Jacobian of
        :math:`\begin{bmatrix}y_1 \\ y_2\end{bmatrix}` with respect to ``param``.
    """

    tensors_ = list[TensorWithJac]()
    for t in tensors:
        if not is_tensor_with_jac(t):
            raise ValueError(
                "Some `jac` fields were not populated. Did you use `autojac.backward` or "
                "`autojac.mtl_backward` before calling `jac_to_grad`?"
            )
        tensors_.append(t)

    if len(tensors_) == 0:
        return

    jacobians = deque(t.jac for t in tensors_)

    if not all([jacobian.shape[0] == jacobians[0].shape[0] for jacobian in jacobians]):
        raise ValueError("All Jacobians should have the same number of rows.")

    if not retain_jac:
        _free_jacs(tensors_)

    if isinstance(aggregator, GramianWeightedAggregator) and not _has_forward_hook(aggregator):
        # When it's possible, avoid the concatenation of the jacobians that can be very costly in
        # memory.
        gradients = _gramian_based(aggregator, jacobians, tensors_)
    else:
        gradients = _jacobian_based(aggregator, jacobians, tensors_)
    accumulate_grads(tensors_, gradients)


def _has_forward_hook(module: nn.Module) -> bool:
    """Return whether the module has any forward hook registered."""
    return len(module._forward_hooks) > 0 or len(module._forward_pre_hooks) > 0


def _jacobian_based(
    aggregator: Aggregator, jacobians: deque[Tensor], tensors: list[TensorWithJac]
) -> list[Tensor]:
    jacobian_matrix = _unite_jacobians(jacobians)
    gradient_vector = aggregator(jacobian_matrix)
    gradients = _disunite_gradient(gradient_vector, tensors)
    return gradients


def _gramian_based(
    aggregator: GramianWeightedAggregator, jacobians: deque[Tensor], tensors: list[TensorWithJac]
) -> list[Tensor]:
    weighting = aggregator.gramian_weighting
    gramian = _compute_gramian_sum(jacobians)
    weights = weighting(gramian)

    gradients = list[Tensor]()
    while jacobians:
        jacobian = jacobians.popleft()  # get jacobian + dereference it to free memory asap
        gradients.append(torch.tensordot(weights, jacobian, dims=1))

    return gradients


def _compute_gramian_sum(jacobians: deque[Tensor]) -> PSDMatrix:
    gramian = sum([compute_gramian(matrix) for matrix in jacobians])
    return cast(PSDMatrix, gramian)


def _unite_jacobians(jacobians: deque[Tensor]) -> Tensor:
    jacobian_matrices = list[Tensor]()
    while jacobians:
        jacobian = jacobians.popleft()  # get jacobian + dereference it to free memory asap
        jacobian_matrices.append(jacobian.reshape(jacobian.shape[0], -1))
    jacobian_matrix = torch.concat(jacobian_matrices, dim=1)
    return jacobian_matrix


def _disunite_gradient(gradient_vector: Tensor, tensors: list[TensorWithJac]) -> list[Tensor]:
    gradient_vectors = gradient_vector.split([t.numel() for t in tensors])
    gradients = [g.view(t.shape) for g, t in zip(gradient_vectors, tensors)]
    return gradients


def _free_jacs(tensors: Iterable[TensorWithJac]) -> None:
    """
    Deletes the ``.jac`` field of the provided tensors.

    :param tensors: The tensors whose ``.jac`` fields should be cleared.
    """

    for t in tensors:
        del t.jac
