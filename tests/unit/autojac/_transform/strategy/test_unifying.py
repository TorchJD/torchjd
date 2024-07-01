import pytest
from unit.autojac._transform.strategy.utils.inputs import aggregator, keys
from unit.autojac._transform.strategy.utils.property_testers import (
    EmptyDictProperty,
    ExpectedStructureProperty,
)

from torchjd.autojac._transform.strategy import UnifyingStrategy


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=keys)])
class TestUnifyingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=[])])
class TestUnifyingEmpty(EmptyDictProperty):
    pass
