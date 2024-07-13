from collections import OrderedDict
from typing import Hashable, Iterable, TypeVar

import torch
from torch import Tensor

from torchjd.aggregation import Aggregator

from ._utils import _OrderedSet, ordered_set
from .base import Transform
from .tensor_dict import EmptyTensorDict, Gradients, GradientVectors, JacobianMatrices, Jacobians

_KeyType = TypeVar("_KeyType", bound=Hashable)
_ValueType = TypeVar("_ValueType")


class Aggregate(Transform[Jacobians, Gradients]):
    def __init__(self, aggregator: Aggregator, key_order: Iterable[Tensor]):
        matrixify = _Matrixify(key_order)
        aggregate_matrices = _AggregateMatrices(aggregator, key_order)
        reshape = _Reshape(key_order)

        self._aggregator_str = str(aggregator)
        self.transform = reshape << aggregate_matrices << matrixify

    def _compute(self, input: Jacobians) -> Gradients:
        return self.transform(input)

    @property
    def required_keys(self) -> set[Tensor]:
        return self.transform.required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self.transform.output_keys


class _AggregateMatrices(Transform[JacobianMatrices, GradientVectors]):
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
        ordered_matrices = self._select_ordered_subdict(jacobian_matrices, self.key_order)
        return self._aggregate_group(ordered_matrices, self.aggregator)

    @property
    def required_keys(self) -> set[Tensor]:
        return set(self.key_order)

    @property
    def output_keys(self) -> set[Tensor]:
        return set(self.key_order)

    @staticmethod
    def _select_ordered_subdict(
        dictionary: dict[_KeyType, _ValueType], ordered_keys: _OrderedSet[_KeyType]
    ) -> OrderedDict[_KeyType, _ValueType]:
        """
        Selects a subset of a dictionary corresponding to the keys given by ``ordered_keys``.
        Returns an OrderedDict in the same order as the provided ``ordered_keys``.
        """

        return OrderedDict([(key, dictionary[key]) for key in ordered_keys])

    @staticmethod
    def _aggregate_group(
        jacobian_matrices: OrderedDict[Tensor, Tensor], aggregator: Aggregator
    ) -> GradientVectors:
        """
        Unites the jacobian matrices and aggregates them using an
        :class:`~torchjd.aggregation.bases.Aggregator`. Returns the obtained gradient vectors.
        """

        if len(jacobian_matrices) == 0:
            return EmptyTensorDict()

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
    ) -> GradientVectors:
        expected_length = sum([matrix.shape[1] for matrix in jacobian_matrices.values()])
        if len(united_gradient_vector) != expected_length:
            raise ValueError(
                "Parameter `united_gradient_vector` should be a vector with length equal to the sum"
                "of the numbers of columns in the jacobian matrices. Found"
                f"`len(united_gradient_vector) = {len(united_gradient_vector)}` and the sum of the "
                f"numbers of columns in the jacobian matrices is {expected_length}."
            )

        gradient_vectors = {}
        start = 0
        for key, jacobian_matrix in jacobian_matrices.items():
            end = start + jacobian_matrix.shape[1]
            current_gradient_vector = united_gradient_vector[start:end]
            gradient_vectors[key] = current_gradient_vector
            start = end
        return GradientVectors(gradient_vectors)


class _Matrixify(Transform[Jacobians, JacobianMatrices]):
    def __init__(self, required_keys: Iterable[Tensor]):
        self._required_keys = set(required_keys)

    def _compute(self, jacobians: Jacobians) -> JacobianMatrices:
        jacobian_matrices = {
            key: jacobian.view(jacobian.shape[0], -1) for key, jacobian in jacobians.items()
        }
        return JacobianMatrices(jacobian_matrices)

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._required_keys


class _Reshape(Transform[GradientVectors, Gradients]):
    def __init__(self, required_keys: Iterable[Tensor]):
        self._required_keys = set(required_keys)

    def _compute(self, gradient_vectors: GradientVectors) -> Gradients:
        gradients = {
            key: gradient_vector.view(key.shape)
            for key, gradient_vector in gradient_vectors.items()
        }
        return Gradients(gradients)

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._required_keys
