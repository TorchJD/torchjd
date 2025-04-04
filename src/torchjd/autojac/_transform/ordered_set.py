from collections import OrderedDict
from typing import Hashable, Iterable, TypeVar

_KeyType = TypeVar("_KeyType", bound=Hashable)


class OrderedSet(OrderedDict[_KeyType, None]):
    """Ordered collection of distinct elements."""

    def __init__(self, elements: Iterable[_KeyType]):
        super().__init__([(element, None) for element in elements])

    def difference_update(self, elements: set[_KeyType]) -> None:
        """Removes all specified elements from the OrderedSet."""

        for element in elements:
            if element in self:
                del self[element]

    def add(self, element: _KeyType) -> None:
        """Adds the specified element to the OrderedSet."""

        self[element] = None
