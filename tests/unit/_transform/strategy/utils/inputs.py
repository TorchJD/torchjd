import math

import torch

from torchjd._transform import JacobianMatrices
from torchjd.aggregation import Random

_param_shapes = [
    [],
    [1],
    [2],
    [5],
    [1, 1],
    [2, 3],
    [5, 5],
    [1, 1, 1],
    [2, 3, 4],
    [5, 5, 5],
    [1, 1, 1, 1],
    [2, 3, 4, 5],
    [5, 5, 5, 5],
]

keys = [torch.zeros(shape) for shape in _param_shapes]


def _make_jacobian_matrices(n_outputs: int) -> JacobianMatrices:
    jacobian_shapes = [[n_outputs, math.prod(shape)] for shape in _param_shapes]
    jacobian_list = [torch.rand(shape) for shape in jacobian_shapes]
    jacobian_matrices = JacobianMatrices({key: jac for key, jac in zip(keys, jacobian_list)})
    return jacobian_matrices


# Fix seed to fix randomness of tensor generation
torch.manual_seed(0)

jacobian_matrix_dicts = [_make_jacobian_matrices(n_outputs) for n_outputs in [1, 2, 5]]

aggregator = Random()
