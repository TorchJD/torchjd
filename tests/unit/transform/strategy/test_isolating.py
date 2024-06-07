import pytest
from unit.transform.strategy.utils.inputs import aggregator, keys
from unit.transform.strategy.utils.property_testers import (
    EmptyDictProperty,
    ExpectedStructureProperty,
)

from torchjd.transform.strategy import IsolatingStrategy


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, keys)])
class TestIsolatingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, [])])
class TestIsolatingEmpty(EmptyDictProperty):
    pass
