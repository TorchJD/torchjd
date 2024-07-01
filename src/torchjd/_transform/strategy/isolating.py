from typing import Iterable

from torch import Tensor

from torchjd._transform.strategy.partitioning import PartitioningStrategy
from torchjd.aggregation import Aggregator


class IsolatingStrategy(PartitioningStrategy):
    """
    TODO: doc
    """

    def __init__(self, aggregator: Aggregator, required_keys: Iterable[Tensor]):
        key_orders = [[key] for key in required_keys]
        aggregators = [aggregator] * len(key_orders)
        super().__init__(key_orders, aggregators)
        self._aggregator = aggregator

    @property
    def aggregator(self) -> Aggregator:
        return self._aggregator

    def __str__(self) -> str:
        return f"Isolating {self.aggregator}"
