import torch

from ._matrix_generation import (
    generate_matrix,
    generate_positively_oriented_matrix,
    generate_stationary_matrix,
)

_matrix_dimension_triples = [
    (1, 1, 1),
    (4, 3, 1),
    (4, 3, 2),
    (4, 3, 3),
    (9, 11, 5),
    (9, 11, 9),
]

_zero_rank_matrix_shapes = [
    (1, 1),
    (4, 3),
    (9, 11),
]

_scales = [0, 1e-25, 1e-10, 1, 1e3, 1e5, 1e10, 1e15, 1e20, 1e25]

# Fix seed to fix randomness of matrix generation
torch.manual_seed(0)

matrices = [
    generate_matrix(n_rows, n_cols, rank) for n_rows, n_cols, rank in _matrix_dimension_triples
]
scaled_matrices = [scale * matrix for scale in _scales for matrix in matrices]
zero_rank_matrices = [torch.zeros([n_rows, n_cols]) for n_rows, n_cols in _zero_rank_matrix_shapes]
matrices_2_plus_rows = [matrix for matrix in matrices + zero_rank_matrices if matrix.shape[0] >= 2]
scaled_matrices_2_plus_rows = [
    matrix for matrix in scaled_matrices + zero_rank_matrices if matrix.shape[0] >= 2
]
matrices_and_triples = [
    generate_positively_oriented_matrix(n_rows, n_cols, rank)
    for n_rows, n_cols, rank in _matrix_dimension_triples
]
stationary_matrices = [
    generate_stationary_matrix(n_rows, n_cols, rank)
    for n_rows, n_cols, rank in _matrix_dimension_triples
]
