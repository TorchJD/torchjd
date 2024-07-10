import pytest

from torchjd.autojac._transform.strategy import ExtrapolatingStrategy

from .utils import EmptyDictProperty, ExpectedStructureProperty, aggregator, keys


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
