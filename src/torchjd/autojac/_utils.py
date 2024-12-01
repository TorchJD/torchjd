from typing import Sequence

from torch import Tensor


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


def _check_retain_graph_compatible_with_chunk_size(
    tensors: list[Tensor],
    retain_graph: bool,
    parallel_chunk_size: int | None,
) -> None:
    tensors_numel = sum([tensor.numel() for tensor in tensors])
    if parallel_chunk_size is not None and parallel_chunk_size < tensors_numel and not retain_graph:
        raise ValueError(
            "When using `retain_graph=False`, parameter `parallel_chunk_size` must be `None` or "
            "large enough to compute all gradients in parallel."
        )


def _get_leaves_of_autograd_graph(roots: list[Tensor], excluded: set[Tensor]) -> set[Tensor]:
    """
    Gets the leaves of the autograd graph of all specified ``tensors``.

    :param roots: Tensors from which the graph traversal should start.
    :param excluded: Tensors excluded from the graph traversal and from the results.
    """

    nodes_to_traverse = [tensor.grad_fn for tensor in roots if tensor not in excluded]
    excluded_nodes = {tensor.grad_fn for tensor in excluded}
    leaves = set()

    # This implementation more or less follows what is advised
    # [here](https://discuss.pytorch.org/t/how-to-access-the-computational-graph/112887), but it is
    # not necessarily robust to future changes, and it's not guaranteed to work.
    # See [this](https://discuss.pytorch.org/t/autograd-graph-traversal/213658) for another question
    # about how to implement this.
    while nodes_to_traverse:
        current_node = nodes_to_traverse.pop()

        if hasattr(current_node, "variable"):
            leaves.add(current_node.variable)
        else:
            nodes_to_traverse += [
                child[0]
                for child in current_node.next_functions
                if child[0] is not None and child[0] not in excluded_nodes
            ]

    return leaves - excluded
