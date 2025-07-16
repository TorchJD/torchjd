import torch
from pytest import mark

from torchjd._autogram._rev_gram_acc import to_sparse_coo_diag, to_sparse_csr_diag


@mark.parametrize(
    ["batch_size", "shape"], [(16, (10,)), (8, (9, 10)), (4, tuple()), (2, (1, 2, 3, 4))]
)
def test_to_sparse_csr_diag(batch_size: int, shape: tuple[int, ...]):
    values = torch.randn((batch_size,) + shape)
    result = to_sparse_csr_diag(values)

    assert result.shape == (batch_size, batch_size) + shape

    for i in range(batch_size):
        assert torch.all(result[i, i] == values[i])


@mark.parametrize(
    ["batch_size", "shape"], [(16, (10,)), (8, (9, 10)), (4, tuple()), (2, (1, 2, 3, 4))]
)
def test_to_sparse_coo_diag(batch_size: int, shape: tuple[int, ...]):
    values = torch.randn((batch_size,) + shape)
    result = to_sparse_coo_diag(values)

    assert result.shape == (batch_size, batch_size) + shape

    for i in range(batch_size):
        assert torch.all(result[i, i] == values[i])
