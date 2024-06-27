from typing import Sequence

from torch import Tensor


def _test_non_negative_optional_chunk_size(parallel_chunk_size: int | None) -> None:
    if not (parallel_chunk_size is None or parallel_chunk_size > 0):
        raise ValueError(
            "`parallel_chunk_size` should be `None` or greater than `0`. (got "
            f"{parallel_chunk_size})"
        )


def _test_retain_graph_compatible_with_chunk_size(
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


def _as_tensor_list(tensors: Sequence[Tensor] | Tensor) -> list[Tensor]:
    if isinstance(tensors, Tensor):
        output = [tensors]
    else:
        output = tensors

    if len(tensors) == 0:
        raise ValueError("`tensors` cannot be an empty iterable of `Tensor`s.")

    return output