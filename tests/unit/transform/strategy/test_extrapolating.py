import pytest
from unit.transform.strategy.utils.inputs import aggregator, keys
from unit.transform.strategy.utils.property_testers import ExpectedStructureProperty

from torchjd.transform.strategy import ExtrapolatingStrategy


@pytest.mark.parametrize(
    "strategy",
    [
        ExtrapolatingStrategy(aggregator, considered_keys=keys, remaining_keys=[]),
        ExtrapolatingStrategy(aggregator, considered_keys=keys[:3], remaining_keys=keys[3:]),
    ],
)
class TestExtrapolating(ExpectedStructureProperty):
    pass
