from collections.abc import Iterable, MutableMapping
from typing import Annotated, Callable

from torch import Node, Tensor
from torch.autograd.graph import get_gradient_edge

Jacobian = Annotated[Callable[[Tensor], tuple[Tensor, ...]], "linear"]


class Derivatives(MutableMapping[Tensor, Tensor]):
    def __init__(self, data: Iterable[tuple[Tensor, Tensor]] = ()):
        self.mapping = {}
        self.update(data)

    def __getitem__(self, key: Tensor) -> Tensor:
        return self.mapping[key]

    def __delitem__(self, key: Tensor) -> None:
        del self.mapping[key]

    def __setitem__(self, key: Tensor, value: Tensor) -> None:
        if key in self:
            self.mapping[key] += value
        self.mapping[key] = value

    def __iter__(self) -> Iterable[Tensor]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.mapping})"


def topological_sort(
    roots: list[Node], included_leaves: set[Node], excluded: set[Node]
) -> list[Node]:
    """
    Returns an ordered list of node in the graph represented by the roots where a node that precede
    another in the graph should precede it in the list.
    This is implemented with Depth-first search (roughly follows
    https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search with the guarantee that
    there is no cycle).
    """

    visited = {node: False for node in excluded}
    reverse_sorted = list()

    def visit(node: Node) -> bool:
        has_leaf = node in included_leaves
        for child, _ in node.next_functions:
            if child is not None:
                if node in visited:
                    has_leaf |= visited[node]
                else:
                    has_leaf |= visit(child)
        visited[node] = has_leaf
        if has_leaf:
            reverse_sorted.append(node)
        return has_leaf

    for root in roots:
        visit(root)

    reverse_sorted.reverse()
    return reverse_sorted


def get_jacobian_and_children(
    node: Node,
) -> tuple[Jacobian, list[Node]]:
    next_functions = [fn[0] for fn in node.next_functions]
    next_functions, indices = zip(
        *[(fn, i) for i, fn in enumerate(next_functions) if fn is not None]
    )

    def jacobian(grad: Tensor) -> tuple[Tensor, ...]:
        output = [t for i, t in enumerate(node(grad)) if i in indices]
        return tuple(output)

    return jacobian, next_functions


def get_node(tensor: Tensor) -> Node:
    return get_gradient_edge(tensor)[0]
