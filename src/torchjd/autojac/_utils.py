from typing import Sequence, Tuple

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


def _traverse_ag_graph(grad_graph: Node) -> list[Tensor]:
    """
    Traverses an autograd graph recursively and extracts all variables (trainable parameters) that lead to the node.
    @param grad_graph: autograd computation node
    @return: found params of calculation
    """
    child_nodes = grad_graph.next_functions
    output_params = []
    for child in child_nodes:
        if child[0] is None:
            # constants in calculation case
            continue
        if hasattr(child[0], "variable"):
            output_params.append(child[0].variable)
        else:
            output_params += _traverse_ag_graph(child[0])
    return output_params


def _determine_shared_not_shared(
    tensor_list: list[list[Tensor]],
) -> Tuple[list[Tensor], list[list[Tensor]]]:
    """
    Based on a list of lists of tensor objects we identify the Tensor objects that are shared in ALL tensor lists
    and the tensor objects that only appear in a tensor list.
    @param tensor_list: list of list containing pytorch Tensor objects
    @return: a tuple containing all shared tensors as a list and a list of list of all individual tensors
    """
    count_map = {}
    for tensors in tensor_list:
        for memory_address in set(
            [id(tensor) for tensor in tensors]
        ):  # we use set to ensure uniqueness
            if memory_address in count_map:
                count_map[memory_address] += 1
            else:
                count_map[memory_address] = 1

    shared_tensors = []
    shared_already_appended = set()
    task_parameters = []
    search_shared = True
    for tensors in tensor_list:
        non_shared_tensors = []
        non_shared_already_appended = set()
        for tensor in tensors:
            memory_address = id(tensor)
            if count_map[memory_address] == len(tensor_list):
                # shared param
                if search_shared:
                    if memory_address not in shared_already_appended:
                        shared_tensors.append(tensor)
                        shared_already_appended.add(memory_address)
            else:
                if memory_address not in non_shared_already_appended:
                    non_shared_tensors.append(tensor)
                    non_shared_already_appended.add(memory_address)
        task_parameters.append(non_shared_tensors)
        search_shared = False  # we already know all shared params after first iteration
    return shared_tensors, task_parameters


def get_tasks_shared_params(losses: list[Tensor]) -> Tuple[list[Tensor], list[list[Tensor]]]:
    """
    Based on a list of pytorch calculations this function identifies the shared parameters and the indvidual parameters
    for each calculation.
    @param losses: list of calculations
    @return: tuple containing all shared tensors as a list and a list of list of all individual tensors
    """
    return _determine_shared_not_shared([_traverse_ag_graph(loss.grad_fn) for loss in losses])
