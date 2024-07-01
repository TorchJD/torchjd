import pytest
from unit._transform.strategy.utils.inputs import aggregator, keys
from unit._transform.strategy.utils.property_testers import (
    EmptyDictProperty,
    ExpectedStructureProperty,
)

from torchjd._transform.strategy import UnifyingStrategy


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=keys)])
class TestUnifyingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=[])])
class TestUnifyingEmpty(EmptyDictProperty):
    pass
