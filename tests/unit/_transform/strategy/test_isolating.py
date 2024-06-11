import pytest
from unit._transform.strategy.utils.inputs import aggregator, keys
from unit._transform.strategy.utils.property_testers import (
    EmptyDictProperty,
    ExpectedStructureProperty,
)

from torchjd._transform.strategy import IsolatingStrategy


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, keys)])
class TestIsolatingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, [])])
class TestIsolatingEmpty(EmptyDictProperty):
    pass
