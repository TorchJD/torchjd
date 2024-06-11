from typing import Iterable, Sequence

from torch import Tensor

from torchjd._transform import Conjunction, Transform
from torchjd._transform._utils import _OrderedSet, ordered_set
from torchjd._transform.strategy.unifying import UnifyingStrategy
from torchjd._transform.subset import Subset
from torchjd._transform.tensor_dict import Gradients, GradientVectors, JacobianMatrices, Jacobians
from torchjd.aggregation import Aggregator


class PartitioningStrategy(Transform[JacobianMatrices, GradientVectors]):
    """
    TODO: doc
    """

    def __init__(
        self,
        key_orders: Sequence[Iterable[Tensor]],
        aggregators: Sequence[Aggregator],
    ):
        key_orders = [ordered_set(key_order) for key_order in key_orders]

        self._check_key_orders_aggregators_same_length(key_orders, aggregators)
        self._check_key_orders_non_empty(key_orders)
        self._check_key_orders_disjoint(key_orders)

        self._required_keys = {key for key_order in key_orders for key in key_order}

        aggregations = []
        for key_order, aggregator in zip(key_orders, aggregators):
            subset = Subset(key_order, self.required_keys)
            strategy = UnifyingStrategy(aggregator, key_order)
            aggregations.append(strategy << subset)

        self.transform = Conjunction(aggregations)

    def _compute(self, jacobians: Jacobians) -> Gradients:
        return self.transform(jacobians)

    @staticmethod
    def _check_key_orders_aggregators_same_length(
        key_orders: Sequence[_OrderedSet[Tensor]], aggregators: Sequence[Aggregator]
    ) -> None:
        if len(key_orders) != len(aggregators):
            raise ValueError(
                "Parameters `key_orders` and `aggregators` should be sequences of the same length. "
                f"Found `len(key_orders) = {len(key_orders)}` and `len(aggregators) = "
                f"{len(aggregators)}`."
            )

    @staticmethod
    def _check_key_orders_non_empty(key_orders: Sequence[_OrderedSet[Tensor]]) -> None:
        for key_order in key_orders:
            if not key_order:
                raise ValueError(
                    "Parameter `key_orders` must be contain non-empty sets. Found a key group "
                    f"`{key_order}`."
                )

    @staticmethod
    def _check_key_orders_disjoint(key_orders: Sequence[_OrderedSet[Tensor]]) -> None:
        all_keys = set()
        for key_order in key_orders:
            if all_keys & key_order.keys():
                raise ValueError(
                    "Parameter `key_orders` must contain disjoint sets. Found `key_orders = "
                    f"{key_orders}`."
                )

    def __str__(self) -> str:
        return "PartitioningStrategy"

    @property
    def required_keys(self) -> set[Tensor]:
        return self._required_keys

    @property
    def output_keys(self) -> set[Tensor]:
        return self._required_keys
