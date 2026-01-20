from pytest import mark
from utils.asserts import assert_psd_matrix
from utils.tensors import randn_

from torchjd._linalg import compute_gramian, is_generalized_matrix


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
