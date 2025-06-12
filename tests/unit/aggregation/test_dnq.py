import torch
from pytest import mark
from torch import Tensor

from torchjd.aggregation._dnq import DNQWrapper
from torchjd.aggregation._upgrad import _UPGrad2

from ._asserts import (
    assert_expected_structure,
    assert_non_differentiable,
    assert_strongly_stationary,
)
from ._inputs import non_strong_matrices, scaled_matrices, typical_matrices

scaled_pairs = [(DNQWrapper(_UPGrad2()), matrix) for matrix in scaled_matrices]
typical_pairs = [(DNQWrapper(_UPGrad2()), matrix) for matrix in typical_matrices]
non_strong_pairs = [(DNQWrapper(_UPGrad2()), matrix) for matrix in non_strong_matrices]
requires_grad_pairs = [(DNQWrapper(_UPGrad2()), torch.ones(3, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: DNQWrapper, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: DNQWrapper, matrix: Tensor):
    assert_strongly_stationary(aggregator, matrix, threshold=5e-03)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: DNQWrapper, matrix: Tensor):
    assert_non_differentiable(aggregator, matrix)
