from collections import OrderedDict
from typing import Iterable

from torch import Tensor

from torchjd._transform import Transform
from torchjd._transform._utils import ordered_set
from torchjd._transform.strategy._utils import _combine_group, _select_ordered_subdict
from torchjd._transform.tensor_dict import GradientVectors, JacobianMatrices
from torchjd.aggregation.bases import _WeightedAggregator


class ExtrapolatingStrategy(Transform[JacobianMatrices, GradientVectors]):
    """
    TODO: doc
    """

    def __init__(
        self,
        aggregator: _WeightedAggregator,
        considered_keys: Iterable[Tensor],
        remaining_keys: Iterable[Tensor],
    ):
        self.considered_keys = ordered_set(considered_keys)
        self.remaining_keys = ordered_set(remaining_keys)
        self.aggregator = aggregator

        key_orders = [self.considered_keys, self.remaining_keys]
        self._required_keys = {key for key_order in key_orders for key in key_order}

    def _compute(self, jacobian_matrices: JacobianMatrices) -> GradientVectors:
        """
        Selects the matrices corresponding to ``considered_keys`` from the provided
        ``jacobian_matrices``. Concatenates them into a single matrix. Aggregates this matrix into
        a gradient vector by using the ``aggregator``. Uses the same weights to combine the
        remaining jacobian matrices (those not corresponding to a key in ``considered_keys``)
        into gradient vectors. Returns all the obtained gradients as a dictionary, with the same
        keys as ``jacobian_matrices``.

        :param jacobian_matrices: The dictionary of jacobian matrices to aggregate. The first
            dimension of each jacobian matrix should be the same.
        """
        considered_matrices = _select_ordered_subdict(jacobian_matrices, self.considered_keys)
        remaining_matrices = _select_ordered_subdict(jacobian_matrices, self.remaining_keys)

        considered_gradient_vectors, gradient_weights = _combine_group(
            considered_matrices, self.aggregator
        )
        remaining_gradient_vectors = self._combine_with_weights(
            remaining_matrices, gradient_weights
        )

        gradient_vectors = considered_gradient_vectors | remaining_gradient_vectors
        return GradientVectors(gradient_vectors)

    @staticmethod
    def _combine_with_weights(
        jacobian_matrices: OrderedDict[Tensor, Tensor], weights: Tensor
    ) -> GradientVectors:
        gradient_vectors = {
            key: _WeightedAggregator.combine(value, weights)
            for key, value in jacobian_matrices.items()
        }
        return GradientVectors(gradient_vectors)

    def __str__(self) -> str:
        return f"Extrapolating {self.aggregator}"

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._required_keys
