from collections import deque

from torch import Tensor
from torch.autograd.graph import GradientEdge, get_gradient_edge


class TargetRegistry:
    """
    Tracks the targets for the second backward phase of the autogram algorithm. Enables computing a
    minimally sufficient subset of leaf targets.
    """

    def __init__(self) -> None:
        self._target_edges: set[GradientEdge] = set()

    def reset(self) -> None:
        self._target_edges: set[GradientEdge] = set()

    def register(self, target: Tensor) -> None:
        """
        Track the GradientEdge of the provided target.

        :param target: Tensor to track.
        """
        self._target_edges.add(get_gradient_edge(target))

    def get_leaf_target_edges(self, excluded: set[GradientEdge]) -> list[GradientEdge]:
        """
        Compute a minimal subset of targets that yields the same differentiation graph traversal:
        the leaf targets. Specifically, this removes targets that are reachable from other targets
        in the differentiation graph, avoiding the need to keep gradients for all targets in memory
        simultaneously.

        :param excluded: Gradient edges that stop graph traversal. Modified in-place.
        :returns: Minimal subset of leaf targets.
        """
        nodes_to_traverse = deque(
            (child, target) for target in self._target_edges for child in _next_edges(target)
        )

        already_added = {child for child, _ in nodes_to_traverse}

        while nodes_to_traverse:
            node, origin = nodes_to_traverse.popleft()
            if node in self._target_edges:
                excluded.add(origin)
            else:
                for child in _next_edges(node):
                    if child not in already_added:
                        nodes_to_traverse.append((child, origin))
                        already_added.add(child)

        return list(self._target_edges - excluded)


def _next_edges(edge: GradientEdge) -> list[GradientEdge]:
    """
    Get the next gradient edges in the differentiation graph from the given edge.

    :param edge: The current gradient edge.
    """
    return [GradientEdge(child, nr) for child, nr in edge.node.next_functions if child is not None]
