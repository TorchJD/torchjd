from pytest import mark
from utils.asserts import assert_psd_matrix
from utils.tensors import randn_

from torchjd._linalg import compute_gramian, is_matrix
from torchjd.aggregation._utils.gramian import normalize, regularize


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
