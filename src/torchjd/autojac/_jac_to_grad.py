from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor

from torchjd.aggregation import Aggregator

from ._accumulation import TensorWithJac, accumulate_grads


def jac_to_grad(params: Iterable[Tensor], aggregator: Aggregator, retain_jac: bool = False) -> None:
    """
    Aggregates the Jacobians stored in the ``.jac`` fields of ``params`` and accumulates the result
    into their ``.grad`` fields.

    :param params: The parameters whose ``.jac`` fields should be aggregated. All Jacobians must
        have the same first dimension (number of outputs).
    :param aggregator: The aggregator used to reduce the Jacobians into gradients.
    :param retain_jac: Whether to preserve the ``.jac`` fields of the parameters.
    """

    params_ = list[TensorWithJac]()
    for p in params:
        if not hasattr(p, "jac"):
            raise ValueError(
                "Some `jac` fields were not populated. Did you use `autojac.backward` before"
                "calling `jac_to_grad`?"
            )
        p_ = cast(TensorWithJac, p)
        params_.append(p_)

    if len(params_) == 0:
        return

    jacobians = [p.jac for p in params_]

    if not all([jacobian.shape[0] == jacobians[0].shape[0] for jacobian in jacobians[1:]]):
        raise ValueError("All Jacobians should have the same number of rows.")

    jacobian_matrix = _unite_jacobians(jacobians)
    gradient_vector = aggregator(jacobian_matrix)
    gradients = _disunite_gradient(gradient_vector, jacobians, params_)
    accumulate_grads(params_, gradients)

    if not retain_jac:
        _free_jacs(params_)


def _unite_jacobians(jacobians: list[Tensor]) -> Tensor:
    jacobian_matrices = [jacobian.reshape(jacobian.shape[0], -1) for jacobian in jacobians]
    jacobian_matrix = torch.concat(jacobian_matrices, dim=1)
    return jacobian_matrix


def _disunite_gradient(
    gradient_vector: Tensor, jacobians: list[Tensor], params: list[TensorWithJac]
) -> list[Tensor]:
    gradient_vectors = []
    start = 0
    for jacobian in jacobians:
        end = start + jacobian[0].numel()
        current_gradient_vector = gradient_vector[start:end]
        gradient_vectors.append(current_gradient_vector)
        start = end
    gradients = [g.view(param.shape) for param, g in zip(params, gradient_vectors, strict=True)]
    return gradients


def _free_jacs(params: Iterable[TensorWithJac]) -> None:
    """
    Deletes the ``.jac`` field of the provided parameters.

    :param params: The parameters whose ``.jac`` fields should be cleared.
    """

    for p in params:
        del p.jac
