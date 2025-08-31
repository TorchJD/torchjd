from collections import deque

from torch.autograd.graph import GradientEdge


class EdgeRegistry:
    """
    Tracks `GradientEdge`s and provides a way to efficiently compute a minimally sufficient subset
    of leaf edges that are reachable from some given `GradientEdge`s.
    """

    def __init__(self) -> None:
        self._edges: set[GradientEdge] = set()

    def reset(self) -> None:
        self._edges = set()

    def register(self, edge: GradientEdge) -> None:
        """
        Track the provided edge.

        :param edge: Edge to track.
        """
        self._edges.add(edge)

    def get_leaf_edges(self, roots: set[GradientEdge]) -> set[GradientEdge]:
        """
        Compute a minimal subset of edges that yields the same differentiation graph traversal: the
        leaf edges. Specifically, this removes edges that are reachable from other edges in the
        differentiation graph, avoiding the need to keep gradients in memory for all edges
        simultaneously.

        :param roots: Roots of the graph traversal. Modified in-place.
        :returns: Minimal subset of leaf edges.
        """

        nodes_to_traverse = deque((child, root) for root in roots for child in _next_edges(root))
        result = {root for root in roots if root in self._edges}

        excluded = roots
        while nodes_to_traverse:
            node, origin = nodes_to_traverse.popleft()
            if node in self._edges:
                result.add(node)
                result.discard(origin)
                origin = node
            for child in _next_edges(node):
                if child not in excluded:
                    nodes_to_traverse.append((child, origin))
                    excluded.add(child)
        return result


def _next_edges(edge: GradientEdge) -> list[GradientEdge]:
    """
    Get the next edges in the autograd graph from the given edge.

    :param edge: The current edge.
    """
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]
