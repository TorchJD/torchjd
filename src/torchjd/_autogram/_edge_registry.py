from collections import deque

from torch import Tensor
from torch.autograd.graph import GradientEdge, get_gradient_edge


class EdgeRegistry:
    """
    Tracks GradientEdges and provides as way to efficiently compute a minimally sufficient subset of
    leaf edges (reaching them passes by all tracked edges).
    """

    def __init__(self) -> None:
        self._edges: set[GradientEdge] = set()

    def reset(self) -> None:
        self._edges = set()

    def register(self, tensor: Tensor) -> None:
        """
        Track the GradientEdge of the provided target tensor.

        :param tensor: Tensor to track.
        """
        self._edges.add(get_gradient_edge(tensor))

    def get_leaf_edges(self, excluded: set[GradientEdge]) -> list[GradientEdge]:
        """
        Compute a minimal subset of edges that yields the same differentiation graph traversal: the
        leaf edges. Specifically, this removes edges that are reachable from other edges in the
        differentiation graph, avoiding the need to keep gradients in memory for all edges
        simultaneously.

        :param excluded: GradientEdges that stop graph traversal. Modified in-place.
        :returns: Minimal subset of leaf edges.
        """
        nodes_to_traverse = deque(
            (child, target) for target in self._edges for child in _next_edges(target)
        )

        already_added = {child for child, _ in nodes_to_traverse}

        while nodes_to_traverse:
            node, origin = nodes_to_traverse.popleft()
            if node in self._edges:
                excluded.add(origin)
            else:
                for child in _next_edges(node):
                    if child not in already_added:
                        nodes_to_traverse.append((child, origin))
                        already_added.add(child)

        return list(self._edges - excluded)


def _next_edges(edge: GradientEdge) -> list[GradientEdge]:
    """
    Get the next GradientEdges in the autograd graph from the given edge.

    :param edge: The current gradient edge.
    """
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]
