import pytest
from unit.transform.strategy.utils.inputs import aggregator, keys
from unit.transform.strategy.utils.property_testers import ExpectedStructureProperty

from torchjd.transform.strategy import IsolatingStrategy


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, keys)])
class TestIsolating(ExpectedStructureProperty):
    pass
