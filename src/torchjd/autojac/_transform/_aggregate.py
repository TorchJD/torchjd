from collections import OrderedDict
from collections.abc import Hashable
from typing import TypeVar

import torch
from torch import Tensor

from torchjd.aggregation import Aggregator

from ._base import RequirementError, TensorDict, Transform
from ._ordered_set import OrderedSet

_KeyType = TypeVar("_KeyType", bound=Hashable)
_ValueType = TypeVar("_ValueType")


class Aggregate(Transform):
    """
    Transform aggregating Jacobians into Gradients.

    It does so by reshaping these Jacobians into matrices, concatenating them into a single matrix,
    applying an aggregator to it, separating the result back into one gradient vector per key, and
    finally reshaping those into gradients of the same shape as their corresponding keys.

    :param aggregator: The aggregator used to aggregate the concatenated jacobian matrix.
    :param key_order: Order in which the different jacobian matrices must be concatenated.
    """

    def __init__(self, aggregator: Aggregator, key_order: OrderedSet[Tensor]):
        matrixify = _Matrixify()
        aggregate_matrices = _AggregateMatrices(aggregator, key_order)
        reshape = _Reshape()

        self._aggregator_str = str(aggregator)
        self.transform = reshape << aggregate_matrices << matrixify

    def __call__(self, input: TensorDict) -> TensorDict:
        return self.transform(input)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        return self.transform.check_keys(input_keys)


class _AggregateMatrices(Transform):
    """
    Transform aggregating JacobiansMatrices into GradientsVectors.

    It does so by concatenating the matrices into a single matrix, applying an aggregator to it and
    separating the result back into one gradient vector per key.

    :param aggregator: The aggregator used to aggregate the concatenated jacobian matrix.
    :param key_order: Order in which the different jacobian matrices must be concatenated.
    """

    def __init__(self, aggregator: Aggregator, key_order: OrderedSet[Tensor]):
        self.key_order = key_order
        self.aggregator = aggregator

    def __call__(self, jacobian_matrices: TensorDict) -> TensorDict:
        """
        Concatenates the provided ``jacobian_matrices`` into a single matrix and aggregates it using
        the ``aggregator``. Returns the dictionary mapping each key from ``jacobian_matrices`` to
        the part of the obtained gradient vector, that corresponds to the jacobian matrix given for
        that key.

        :param jacobian_matrices: The dictionary of jacobian matrices to aggregate. The first
            dimension of each jacobian matrix should be the same.
        """
        ordered_matrices = self._select_ordered_subdict(jacobian_matrices, self.key_order)
        return self._aggregate_group(ordered_matrices, self.aggregator)

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        if not set(self.key_order) == input_keys:
            raise RequirementError(
                f"The input_keys must match the key_order. Found input_keys {input_keys} and"
                f"key_order {self.key_order}."
            )
        return input_keys

    @staticmethod
    def _select_ordered_subdict(
        dictionary: dict[_KeyType, _ValueType], ordered_keys: OrderedSet[_KeyType]
    ) -> OrderedDict[_KeyType, _ValueType]:
        """
        Selects a subset of a dictionary corresponding to the keys given by ``ordered_keys``.
        Returns an OrderedDict in the same order as the provided ``ordered_keys``.
        """

        return OrderedDict([(key, dictionary[key]) for key in ordered_keys])

    @staticmethod
    def _aggregate_group(
        jacobian_matrices: OrderedDict[Tensor, Tensor], aggregator: Aggregator
    ) -> TensorDict:
        """
        Unites the jacobian matrices and aggregates them using an
        :class:`~torchjd.aggregation._aggregator_bases.Aggregator`. Returns the obtained gradient
        vectors.
        """

        if len(jacobian_matrices) == 0:
            return {}

        united_jacobian_matrix = _AggregateMatrices._unite(jacobian_matrices)
        united_gradient_vector = aggregator(united_jacobian_matrix)
        gradient_vectors = _AggregateMatrices._disunite(united_gradient_vector, jacobian_matrices)
        return gradient_vectors

    @staticmethod
    def _unite(jacobian_matrices: OrderedDict[Tensor, Tensor]) -> Tensor:
        return torch.cat(list(jacobian_matrices.values()), dim=1)

    @staticmethod
    def _disunite(
        united_gradient_vector: Tensor, jacobian_matrices: OrderedDict[Tensor, Tensor]
    ) -> TensorDict:
        gradient_vectors = {}
        start = 0
        for key, jacobian_matrix in jacobian_matrices.items():
            end = start + jacobian_matrix.shape[1]
            current_gradient_vector = united_gradient_vector[start:end]
            gradient_vectors[key] = current_gradient_vector
            start = end
        return gradient_vectors


class _Matrixify(Transform):
    """Transform reshaping Jacobians into JacobianMatrices."""

    def __call__(self, jacobians: TensorDict) -> TensorDict:
        jacobian_matrices = {
            key: jacobian.view(jacobian.shape[0], -1) for key, jacobian in jacobians.items()
        }
        return jacobian_matrices

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        return input_keys


class _Reshape(Transform):
    """Transform reshaping GradientVectors into Gradients."""

    def __call__(self, gradient_vectors: TensorDict) -> TensorDict:
        gradients = {
            key: gradient_vector.view(key.shape)
            for key, gradient_vector in gradient_vectors.items()
        }
        return gradients

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        return input_keys
