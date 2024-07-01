import pytest
from unit.autojac._transform.strategy.utils.inputs import aggregator, keys
from unit.autojac._transform.strategy.utils.property_testers import (
    EmptyDictProperty,
    ExpectedStructureProperty,
)

from torchjd.autojac._transform.strategy import PartitioningStrategy


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
