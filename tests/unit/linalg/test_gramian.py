from pytest import mark
from utils.asserts import assert_psd_matrix
from utils.tensors import randn_

from torchjd._linalg._gramian import compute_gramian


@mark.parametrize(
    "shape",
    [
        [3, 1],
        [4, 4],
        [4, 3],
        [6, 7],
    ],
)
def test_gramian_is_psd(shape: list[int]):
    matrix = randn_(shape)
    gramian = compute_gramian(matrix)
    assert_psd_matrix(gramian)
