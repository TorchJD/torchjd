from collections import OrderedDict
from typing import Hashable, Iterable, TypeVar

import torch
from torch import Tensor

from torchjd.aggregation import Aggregator

from .._utils import _OrderedSet, ordered_set
from ..base import Transform
from ..tensor_dict import EmptyTensorDict, GradientVectors, JacobianMatrices


class UnifyingStrategy(Transform[JacobianMatrices, GradientVectors]):
    """
    TODO: doc
    """

    def __init__(self, aggregator: Aggregator, key_order: Iterable[Tensor]):
        self.key_order = ordered_set(key_order)
        self.aggregator = aggregator

    def _compute(self, jacobian_matrices: JacobianMatrices) -> GradientVectors:
        """
        Concatenates the provided ``jacobian_matrices`` into a single matrix and aggregates it using
        the ``aggregator``. Returns the dictionary mapping each key from ``jacobian_matrices`` to
        the part of the obtained gradient vector, that corresponds to the jacobian matrix given for
        that key.

        :param jacobian_matrices: The dictionary of jacobian matrices to aggregate. The first
            dimension of each jacobian matrix should be the same.
        """
        ordered_matrices = _select_ordered_subdict(jacobian_matrices, self.key_order)
        return _aggregate_group(ordered_matrices, self.aggregator)

    def __str__(self) -> str:
        return f"Unifying {self.aggregator}"

    @property
    def required_keys(self) -> set[Tensor]:
        return set(self.key_order)

    @property
    def output_keys(self) -> set[Tensor]:
        return set(self.key_order)


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
