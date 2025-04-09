from collections import OrderedDict
from collections.abc import Set
from typing import Hashable, Iterable, TypeVar

_KeyType = TypeVar("_KeyType", bound=Hashable)


class OrderedSet(OrderedDict[_KeyType, None], Set[_KeyType]):
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

    def __add__(self, other: "OrderedSet[_KeyType]") -> "OrderedSet[_KeyType]":
        """Creates a new OrderedSet with the elements of self followed by the elements of other."""

        return OrderedSet([*self, *other])
