import torch
from pytest import mark
from torch.testing import assert_close
from utils.tensors import randn_, zeros_

from torchjd.autogram.diagonal_sparse_tensor import _POINTWISE_FUNCTIONS, DiagonalSparseTensor


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
    a = randn_(shape)
    b = DiagonalSparseTensor(a, list(range(len(shape))))

    assert_close(a, b)


@mark.parametrize("dim", [1, 2, 3, 4, 5, 10])
def test_diag_equivalence(dim: int):
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [0, 0])

    diag_a = torch.diag(a)

    assert_close(b, diag_a)


def test_three_virtual_single_physical():
    dim = 10
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [0, 0, 0])

    expected = zeros_([dim, dim, dim])
    for i in range(dim):
        expected[i, i, i] = a[i]

    assert_close(b, expected)


@mark.parametrize("func", _POINTWISE_FUNCTIONS)
def test_pointwise(func):
    dim = 10
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [0, 0])
    c = b.to_dense()
    res = func(b)
    assert isinstance(res, DiagonalSparseTensor)

    assert_close(res, func(c), equal_nan=True)


@mark.parametrize("func", [torch.mean, torch.sum])
def test_unary(func):
    dim = 10
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [0, 0])
    c = b.to_dense()

    res = func(b)
    assert_close(res, func(c))
