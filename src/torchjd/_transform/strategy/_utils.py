from collections import OrderedDict
from typing import Hashable, TypeVar

import torch
from torch import Tensor

from torchjd._transform._utils import _OrderedSet
from torchjd._transform.tensor_dict import EmptyTensorDict, GradientVectors
from torchjd.aggregation.bases import Aggregator, _WeightedAggregator

_KeyType = TypeVar("_KeyType", bound=Hashable)
_ValueType = TypeVar("_ValueType")


def _select_ordered_subdict(
    dictionary: dict[_KeyType, _ValueType], ordered_keys: _OrderedSet[_KeyType]
) -> OrderedDict[_KeyType, _ValueType]:
    """
    Selects a subset of a dictionary corresponding to the keys given by ``ordered_keys``.
    Returns an OrderedDict in the same order as the provided ``ordered_keys``.
    """

    return OrderedDict([(key, dictionary[key]) for key in ordered_keys])


def _aggregate_group(
    jacobian_matrices: OrderedDict[Tensor, Tensor], aggregator: Aggregator
) -> GradientVectors:
    """
    Unites the jacobian matrices and aggregates them using an
    :class:`~torchjd.aggregation.bases.Aggregator`. Returns the obtained gradient vectors.
    """

    if len(jacobian_matrices) == 0:
        return EmptyTensorDict()

    united_jacobian_matrix = _unite(jacobian_matrices)
    united_gradient_vector = aggregator(united_jacobian_matrix)
    gradient_vectors = _disunite(united_gradient_vector, jacobian_matrices)
    return gradient_vectors


def _combine_group(
    jacobian_matrices: OrderedDict[Tensor, Tensor],
    aggregator: _WeightedAggregator,
) -> tuple[GradientVectors, Tensor]:
    """
    Unites the jacobian matrices and aggregates them using a
    :class:`~torchjd.aggregation.bases.WeightedAggregator`. Returns the obtained gradient
    vectors and the associated weights.
    """

    if len(jacobian_matrices) == 0:
        return EmptyTensorDict(), torch.empty([0])

    united_jacobian_matrix = _unite(jacobian_matrices)
    gradient_weights = aggregator.weighting(united_jacobian_matrix)
    united_gradient_vector = aggregator.combine(united_jacobian_matrix, gradient_weights)
    gradient_vectors = _disunite(united_gradient_vector, jacobian_matrices)
    return gradient_vectors, gradient_weights


def _unite(jacobian_matrices: OrderedDict[Tensor, Tensor]) -> Tensor:
    return torch.cat(list(jacobian_matrices.values()), dim=1)


def _disunite(
    united_gradient_vector: Tensor, jacobian_matrices: OrderedDict[Tensor, Tensor]
) -> GradientVectors:
    expected_length = sum([matrix.shape[1] for matrix in jacobian_matrices.values()])
    if len(united_gradient_vector) != expected_length:
        raise ValueError(
            "Parameter `united_gradient_vector` should be a vector with length equal to the sum of "
            "the numbers of columns in the jacobian matrices. Found `len(united_gradient_vector) = "
            f"{len(united_gradient_vector)}` and the sum of the numbers of columns in the jacobian "
            f"matrices is {expected_length}."
        )

    gradient_vectors = {}
    start = 0
    for key, jacobian_matrix in jacobian_matrices.items():
        end = start + jacobian_matrix.shape[1]
        current_gradient_vector = united_gradient_vector[start:end]
        gradient_vectors[key] = current_gradient_vector
        start = end
    return GradientVectors(gradient_vectors)
