import torch
from pytest import mark

from torchjd.sparse._linalg import hnf_decomposition


@mark.parametrize(
    ["shape", "max_rank"],
    [
        ([5, 7], 3),
        ([1, 7], 1),
        ([5, 1], 1),
        ([7, 5], 2),
        ([5, 7], 5),
        ([7, 5], 5),
    ],
)
def test_hnf_decomposition(shape: tuple[int, int], max_rank: int):
    # Generate a matrix A of desired shape and rank max_rank with high probability and lower
    # otherwise.
    U = torch.randint(-50, 51, [shape[0], max_rank], dtype=torch.int64)
    V = torch.randint(-50, 51, [max_rank, shape[1]], dtype=torch.int64)
    A = U @ V
    H, U, V = hnf_decomposition(A)

    rank = H.shape[1]

    assert rank <= max_rank
    assert torch.equal(V @ U, torch.eye(rank, dtype=torch.int64))
    assert torch.equal(H @ V, A)
    assert torch.equal(A @ U, H)
