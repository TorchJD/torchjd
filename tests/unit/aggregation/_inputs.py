import torch
from unit.conftest import DEVICE

from ._matrix_samplers import NonWeakSampler, NormalSampler, StrictlyWeakSampler, StrongSampler

_normal_dims = [
    (1, 1, 1),
    (4, 3, 1),
    (4, 3, 2),
    (4, 3, 3),
    (9, 11, 5),
    (9, 11, 9),
]

_zero_dims = [
    (1, 1, 0),
    (4, 3, 0),
    (9, 11, 0),
]

_stationarity_dims = [
    (20, 10, 10),
    (20, 10, 5),
    (20, 10, 1),
    (20, 100, 1),
    (20, 100, 19),
]

_scales = [0.0, 1e-10, 1e3, 1e5, 1e10, 1e15]

_rng = torch.Generator(device=DEVICE).manual_seed(0)

matrices = [NormalSampler(m, n, r)(_rng) for m, n, r in _normal_dims]
zero_matrices = [torch.zeros([m, n]) for m, n, _ in _zero_dims]
strong_matrices = [StrongSampler(m, n, r)(_rng) for m, n, r in _stationarity_dims]
strictly_weak_matrices = [StrictlyWeakSampler(m, n, r)(_rng) for m, n, r in _stationarity_dims]
non_weak_matrices = [NonWeakSampler(m, n, r)(_rng) for m, n, r in _stationarity_dims]

scaled_matrices = [scale * matrix for scale in _scales for matrix in matrices]

non_strong_matrices = strictly_weak_matrices + non_weak_matrices
typical_matrices = zero_matrices + matrices + strong_matrices + non_strong_matrices

scaled_matrices_2_plus_rows = [matrix for matrix in scaled_matrices if matrix.shape[0] >= 2]
typical_matrices_2_plus_rows = [matrix for matrix in typical_matrices if matrix.shape[0] >= 2]

# It seems that NashMTL does not work for matrices with 1 row, so we make different matrices for it.
_nashmtl_dims = [
    (3, 1, 1),
    (4, 3, 1),
    (4, 3, 2),
    (4, 3, 3),
    (9, 11, 5),
    (9, 11, 9),
]
nash_mtl_matrices = [NormalSampler(m, n, r)(_rng) for m, n, r in _nashmtl_dims]

_dnq_upgrad_dims = [
    (2, 1, 1),
    (2, 2, 2),
    (4, 100, 4),
    (8, 100, 8),
    (16, 100, 16),
    (32, 100, 32),
    (64, 100, 64),
    (128, 100, 100),
    (16, 100, 16),
    (32, 100, 32),
    (64, 100, 64),
    (128, 100, 100),
    (256, 100, 100),
    (512, 100, 100),
    (1024, 100, 100),
    (2048, 100, 100),
    (4096, 100, 100),
    (8192, 100, 100),
    (16384, 100, 100),
]
dnq_upgrad_matrices = [NormalSampler(m, n, r)(_rng) for m, n, r in _dnq_upgrad_dims]
