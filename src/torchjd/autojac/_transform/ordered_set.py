from collections import OrderedDict
from typing import Iterable, TypeAlias

from torchjd.autojac._transform._utils import _KeyType

_OrderedSet: TypeAlias = OrderedDict[_KeyType, None]


def ordered_set(elements: Iterable[_KeyType]) -> _OrderedSet[_KeyType]:
    return OrderedDict.fromkeys(elements, None)
