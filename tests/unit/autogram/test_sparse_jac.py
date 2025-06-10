import torch
from torch import Tensor, vmap
from torch.func import vjp


def test_sparse_jac():
    m = 1000
    x_ = torch.ones(m, requires_grad=True)

    def func(x__):
        return x__ * 2

    I_coo = get_sparse_identity_coo(m)
    I_bsr = get_sparse_identity_bsr(10, 100)

    print(I_bsr)
    print(I_coo[0:10])

    # print(I_bsr.values().shape)
    jacobian = vmap(vjp(func, x_)[1])(I_coo)[0]

    M = torch.randn((2, m))

    R = M @ jacobian

    print(R)


def get_sparse_identity_coo(m: int) -> Tensor:
    indices = torch.arange(0, m).expand((2, m))
    values = torch.full((m,), 1.0)
    S = torch.sparse_coo_tensor(indices, values)
    return S


def get_sparse_identity_bsr(block_size: int, n_blocks: int) -> Tensor:
    matrix_dim = block_size * n_blocks

    # crow_indices: Each row block has one non-zero block (on the diagonal)
    # The length will be n_blocks + 1, as crow_indices[i] gives the start index
    # for the i-th row block's non-zero column blocks in col_indices.
    # Since each row block has exactly one block, it's a simple range.
    crow_indices = torch.arange(0, n_blocks + 1)

    # col_indices: For an identity matrix, the non-zero blocks are on the diagonal.
    # So, the column index for each block is the same as its row index.
    col_indices = torch.arange(0, n_blocks)

    # values: Each block is an identity matrix of size block_size x block_size.
    # We need n_blocks such identity matrices.
    identity_block = torch.eye(block_size, block_size)
    values = identity_block.repeat(n_blocks, 1, 1)

    # Create the block-sparse BSR tensor
    S = torch.sparse_bsr_tensor(
        crow_indices,
        col_indices,
        values,
        size=(matrix_dim, matrix_dim),
        dtype=torch.float32,  # Default dtype, can be changed if needed
    )
    return S


def get_sparse_identity_bsc(block_size: int, n_blocks: int) -> Tensor:
    # Total number of rows/columns in the dense matrix
    matrix_dim = block_size * n_blocks

    # crow_indices: Each row block has one non-zero block (on the diagonal)
    # The length will be n_blocks + 1, as crow_indices[i] gives the start index
    # for the i-th row block's non-zero column blocks in col_indices.
    # Since each row block has exactly one block, it's a simple range.
    crow_indices = torch.arange(0, n_blocks + 1)

    # col_indices: For an identity matrix, the non-zero blocks are on the diagonal.
    # So, the column index for each block is the same as its row index.
    col_indices = torch.arange(0, n_blocks)

    # values: Each block is an identity matrix of size block_size x block_size.
    # We need n_blocks such identity matrices.
    identity_block = torch.eye(block_size, block_size)
    values = identity_block.repeat(n_blocks, 1, 1)

    # Create the block-sparse BSR tensor
    S = torch.sparse_bsr_tensor(
        crow_indices,
        col_indices,
        values,
        size=(matrix_dim, matrix_dim),
        dtype=torch.float32,  # Default dtype, can be changed if needed
    )
    return S
