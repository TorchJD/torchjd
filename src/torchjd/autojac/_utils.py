from collections import deque
from typing import Iterable, Sequence

from torch import Tensor
from torch.autograd.graph import Node


def _check_optional_positive_chunk_size(parallel_chunk_size: int | None) -> None:
    if not (parallel_chunk_size is None or parallel_chunk_size > 0):
        raise ValueError(
            "`parallel_chunk_size` should be `None` or greater than `0`. (got "
            f"{parallel_chunk_size})"
        )


def _as_tensor_list(tensors: Sequence[Tensor] | Tensor) -> list[Tensor]:
    if isinstance(tensors, Tensor):
        output = [tensors]
    else:
        output = list(tensors)
    return output


def _get_leaf_tensors(tensors: Iterable[Tensor], excluded: Iterable[Tensor]) -> set[Tensor]:
    """
    Gets the leaves of the autograd graph of all specified ``tensors``.

    :param tensors: Tensors from which the graph traversal should start. They should all require
        grad and not be leaves.
    :param excluded: Tensors whose grad_fn should be excluded from the graph traversal. They should
        all require grad and not be leaves.

    """

    if any([tensor.grad_fn is None for tensor in tensors]):
        raise ValueError("All `tensors` should have a `grad_fn`.")

    if any([tensor.grad_fn is None for tensor in excluded]):
        raise ValueError("All `excluded` tensors should have a `grad_fn`.")

    accumulate_grads = _get_descendant_accumulate_grads(
        roots={tensor.grad_fn for tensor in tensors},
        excluded_nodes={tensor.grad_fn for tensor in excluded},
    )
    leaves = {g.variable for g in accumulate_grads}

    return leaves


def _get_descendant_accumulate_grads(roots: set[Node], excluded_nodes: set[Node]) -> set[Node]:
    """
    Gets the AccumulateGrad descendants of the specified nodes.

    :param roots: Root nodes from which the graph traversal should start.
    :param excluded_nodes: Nodes excluded from the graph traversal.
    """

    excluded_nodes = set(excluded_nodes)  # Re-instantiate set to avoid modifying input
    result = set()
    nodes_to_traverse = deque(roots - excluded_nodes)

    # This implementation more or less follows what is advised in
    # https://discuss.pytorch.org/t/autograd-graph-traversal/213658 and what was suggested in
    # https://github.com/TorchJD/torchjd/issues/216.
    while nodes_to_traverse:
        node = nodes_to_traverse.popleft()  # Breadth-first

        if node.__class__.__name__ == "AccumulateGrad":
            result.add(node)

        for child, _ in node.next_functions:
            if child is not None and child not in excluded_nodes:
                nodes_to_traverse.append(child)  # Append to the right
                excluded_nodes.add(child)

    return result
