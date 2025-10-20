import torch
from pytest import mark
from torch.testing import assert_close

from torchjd.autogram.diagonal_sparse_tensor import DiagonalSparseTensor


@mark.parametrize(
    "shape",
    [
        [],
        [1],
        [3],
        [1, 1],
        [1, 4],
        [3, 1],
        [1, 2, 3],
    ],
)
def test_diagonal_spase_tensor_scalar(shape: list[int]):
    a = torch.randn(shape)
    b = DiagonalSparseTensor(a, list(range(len(shape))))

    assert_close(a, b)


@mark.parametrize("dim", [1, 2, 3, 4, 5, 10])
def test_diag_equivalence(dim: int):
    a = torch.randn([dim])
    b = DiagonalSparseTensor(a, [0, 0])

    diag_a = torch.diag(a)

    assert_close(b, diag_a)
