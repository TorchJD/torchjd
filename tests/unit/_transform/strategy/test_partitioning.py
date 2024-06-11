import pytest
from unit._transform.strategy.utils.inputs import aggregator, keys
from unit._transform.strategy.utils.property_testers import (
    EmptyDictProperty,
    ExpectedStructureProperty,
)

from torchjd._transform.strategy import PartitioningStrategy


@pytest.mark.parametrize(
    "strategy",
    [
        PartitioningStrategy(
            key_orders=[keys[:3], keys[3:]],
            aggregators=[aggregator, aggregator],
        ),
    ],
)
class TestPartitioningStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [PartitioningStrategy([], [])])
class TestPartitioningEmpty(EmptyDictProperty):
    pass
