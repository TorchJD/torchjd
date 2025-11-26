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
@mark.parametrize("reduced", [True, False])
def test_hnf_decomposition(shape: tuple[int, int], max_rank: int, reduced: bool):
    # Generate a matrix A of desired shape and rank max_rank with high probability and lower
    # otherwise.
    U = torch.randint(-10, 11, [shape[0], max_rank], dtype=torch.int64)
    V = torch.randint(-10, 11, [max_rank, shape[1]], dtype=torch.int64)
    A = U @ V
    H, U, V = hnf_decomposition(A, reduced)

    r = H.shape[1]

    # Note that with these assert, the rank is typically correct as it is at most max_rank, which it
    # is with high probability, and we can reconstruct A=H @ V, so the rank of H is at least that of
    # A, similarly, the rank of H is at most that of A.
    if reduced:
        assert r <= max_rank
    else:
        assert torch.equal(U @ V, torch.eye(r, dtype=torch.int64))
    assert torch.equal(V @ U, torch.eye(r, dtype=torch.int64))
    assert torch.equal(H @ V, A)
    assert torch.equal(A @ U, H)

    # Check H is lower triangular (its upper triangle must be zero)
    assert torch.equal(torch.triu(H, diagonal=1), torch.zeros_like(H))
