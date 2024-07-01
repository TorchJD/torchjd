import pytest
from unit.autojac._transform.strategy.utils import (
    EmptyDictProperty,
    ExpectedStructureProperty,
    aggregator,
    keys,
)

from torchjd.autojac._transform.strategy import UnifyingStrategy


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=keys)])
class TestUnifyingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=[])])
class TestUnifyingEmpty(EmptyDictProperty):
    pass
