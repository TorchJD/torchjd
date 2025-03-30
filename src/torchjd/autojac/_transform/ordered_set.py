from collections import OrderedDict
from typing import Iterable

from torchjd.autojac._transform._utils import _KeyType


class OrderedSet(OrderedDict[_KeyType, None]):
    """
    Collection representing a set whose order is preserved at construction and whose order
    matters in comparisons.
    """

    def __init__(self, elements: Iterable[_KeyType]):
        super().__init__([(element, None) for element in elements])
