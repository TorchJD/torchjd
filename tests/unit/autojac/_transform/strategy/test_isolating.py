import pytest

from torchjd.autojac._transform.strategy import IsolatingStrategy

from .utils import EmptyDictProperty, ExpectedStructureProperty, aggregator, keys


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, keys)])
class TestIsolatingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [IsolatingStrategy(aggregator, [])])
class TestIsolatingEmpty(EmptyDictProperty):
    pass
