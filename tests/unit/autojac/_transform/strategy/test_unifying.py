import pytest

from torchjd.autojac._transform.strategy import UnifyingStrategy

from .utils import EmptyDictProperty, ExpectedStructureProperty, aggregator, keys


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=keys)])
class TestUnifyingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=[])])
class TestUnifyingEmpty(EmptyDictProperty):
    pass
