import pytest
from unit.transform.strategy.utils.inputs import aggregator, keys
from unit.transform.strategy.utils.property_testers import ExpectedStructureProperty

from torchjd.transform.strategy import PartitioningStrategy


@pytest.mark.parametrize(
    "strategy",
    [
        PartitioningStrategy(
            key_orders=[keys[:3], keys[3:]],
            aggregators=[aggregator, aggregator],
        ),
    ],
)
class TestPartitioning(ExpectedStructureProperty):
    pass
