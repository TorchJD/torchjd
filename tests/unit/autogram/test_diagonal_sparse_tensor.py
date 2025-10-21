import torch
from pytest import mark
from torch.testing import assert_close
from utils.tensors import randn_, zeros_

from torchjd.autogram.diagonal_sparse_tensor import DiagonalSparseTensor, _pointwise_functions


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


@mark.parametrize("func", _pointwise_functions)
def test_pointwise(func):
    dim = 100
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [0, 0])
    c = b.to_dense()
    d = func(b)
    assert isinstance(d, DiagonalSparseTensor)

    # need to be careful about nans
    assert_close(d, func(c))
