import pytest

from torchjd.autojac._transform.strategy import PartitioningStrategy

from .utils import EmptyDictProperty, ExpectedStructureProperty, aggregator, keys


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
