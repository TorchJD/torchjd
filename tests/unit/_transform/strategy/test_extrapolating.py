import pytest
from unit._transform.strategy.utils.inputs import aggregator, keys
from unit._transform.strategy.utils.property_testers import (
    EmptyDictProperty,
    ExpectedStructureProperty,
)

from torchjd._transform.strategy import ExtrapolatingStrategy


@pytest.mark.parametrize(
    "strategy",
    [
        ExtrapolatingStrategy(aggregator, considered_keys=keys, remaining_keys=[]),
        ExtrapolatingStrategy(aggregator, considered_keys=keys[:3], remaining_keys=keys[3:]),
    ],
)
class TestExtrapolatingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [ExtrapolatingStrategy(aggregator, [], [])])
class TestExtrapolatingEmpty(EmptyDictProperty):
    pass
