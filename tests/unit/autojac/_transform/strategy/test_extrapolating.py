import pytest
from unit.autojac._transform.strategy.utils import (
    EmptyDictProperty,
    ExpectedStructureProperty,
    aggregator,
    keys,
)

from torchjd.autojac._transform.strategy import ExtrapolatingStrategy


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
