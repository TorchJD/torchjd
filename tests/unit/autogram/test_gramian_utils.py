import torch
from pytest import mark
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import randn_

from torchjd.autogram._gramian_utils import movedim_gramian, reshape_gramian


def _compute_gramian(matrix: Tensor) -> Tensor:
    """Contracts the last dimension of matrix to make it into a Gramian."""

    indices = list(range(matrix.ndim))
    transposed_matrix = matrix.movedim(indices, indices[::-1])
    return torch.tensordot(matrix, transposed_matrix, dims=([-1], [0]))


@mark.parametrize(
    ["original_shape", "target_shape"],
    [
        ([], []),
        ([], [1, 1]),
        ([1], []),
        ([12], [2, 3, 2]),
        ([12], [4, 3]),
        ([12], [12]),
        ([4, 3], [12]),
        ([4, 3], [2, 3, 2]),
        ([4, 3], [3, 4]),
        ([4, 3], [4, 3]),
        ([6, 7, 9], [378]),
        ([6, 7, 9], [9, 42]),
        ([6, 7, 9], [2, 7, 27]),
        ([6, 7, 9], [6, 7, 9]),
    ],
)
def test_reshape_gramian(original_shape: list[int], target_shape: list[int]):
    """Tests that reshape_gramian is such that _compute_gramian is equivariant to a reshape."""

    original_matrix = randn_(original_shape + [2])
    target_matrix = original_matrix.reshape(target_shape + [2])

    original_gramian = _compute_gramian(original_matrix)
    target_gramian = _compute_gramian(target_matrix)

    reshaped_gramian = reshape_gramian(original_gramian, target_shape)

    assert_close(reshaped_gramian, target_gramian)


@mark.parametrize(
    ["shape", "source", "destination"],
    [
        ([], [], []),
        ([1], [0], [0]),
        ([1], [], []),
        ([1, 1], [], []),
        ([1, 1], [1], [0]),
        ([6, 7], [1], [0]),
        ([3, 1], [0, 1], [1, 0]),
        ([1, 1, 1], [], []),
        ([3, 2, 5], [], []),
        ([1, 1, 1], [2], [0]),
        ([3, 2, 5], [1], [2]),
        ([2, 2, 3], [0, 2], [1, 0]),
        ([2, 2, 3], [0, 2, 1], [1, 0, 2]),
    ],
)
def test_movedim_gramian(shape: list[int], source: list[int], destination: list[int]):
    """Tests that movedim_gramian is such that _compute_gramian is equivariant to a movedim."""

    original_matrix = randn_(shape + [2])
    target_matrix = original_matrix.movedim(source, destination)

    original_gramian = _compute_gramian(original_matrix)
    target_gramian = _compute_gramian(target_matrix)

    moveddim_gramian = movedim_gramian(original_gramian, source, destination)

    assert_close(moveddim_gramian, target_gramian)
