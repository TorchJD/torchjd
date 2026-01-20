from pytest import mark
from utils.asserts import assert_psd_matrix
from utils.tensors import randn_

from torchjd._linalg import compute_gramian, is_generalized_matrix, is_matrix
from torchjd._linalg._gramian import normalize, regularize


@mark.parametrize(
    "shape",
    [
        [1],
        [12],
        [4, 3],
        [3, 1],
        [4, 4],
        [4, 3],
        [6, 7],
        [5, 0],
        [6, 7, 9],
    ],
)
def test_gramian_is_psd(shape: list[int]):
    matrix = randn_(shape)
    assert is_generalized_matrix(matrix)
    gramian = compute_gramian(matrix)
    assert_psd_matrix(gramian)


@mark.parametrize(
    "shape",
    [
        [3, 1],
        [4, 4],
        [4, 3],
        [6, 7],
        [5, 0],
    ],
)
def test_normalize_yields_psd(shape: list[int]):
    matrix = randn_(shape)
    assert is_matrix(matrix)
    gramian = compute_gramian(matrix)
    normalized_gramian = normalize(gramian, 1e-05)
    assert_psd_matrix(normalized_gramian)


@mark.parametrize(
    "shape",
    [
        [3, 1],
        [4, 4],
        [4, 3],
        [6, 7],
        [5, 0],
    ],
)
def test_regularize_yields_psd(shape: list[int]):
    matrix = randn_(shape)
    assert is_matrix(matrix)
    gramian = compute_gramian(matrix)
    normalized_gramian = regularize(gramian, 1e-05)
    assert_psd_matrix(normalized_gramian)
