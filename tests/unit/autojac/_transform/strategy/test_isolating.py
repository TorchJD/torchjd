import pytest
from unit.autojac._transform.strategy.utils.inputs import aggregator, keys
from unit.autojac._transform.strategy.utils.property_testers import (
    EmptyDictProperty,
    ExpectedStructureProperty,
)

from torchjd.autojac._transform.strategy import IsolatingStrategy


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, keys)])
class TestIsolatingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, [])])
class TestIsolatingEmpty(EmptyDictProperty):
    pass
