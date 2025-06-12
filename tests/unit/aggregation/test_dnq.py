import torch
from pytest import mark
from torch import Tensor

from torchjd.aggregation._dnq import DNQWrapper
from torchjd.aggregation._upgrad import _UPGrad2

from ._asserts import (
    assert_expected_structure,
    assert_linear_under_scaling,
    assert_non_conflicting,
    assert_non_differentiable,
    assert_strongly_stationary,
)
from ._inputs import dnq_matrices, non_strong_matrices, scaled_matrices, typical_matrices


def is_ok(m):
    # m is power of two and >= 2
    return m > 1 and (m & (m - 1)) == 0


scaled_pairs = [
    (DNQWrapper(_UPGrad2()), matrix) for matrix in scaled_matrices if is_ok(matrix.shape[0])
]
typical_pairs = [
    (DNQWrapper(_UPGrad2()), matrix)
    for matrix in typical_matrices + dnq_matrices
    if is_ok(matrix.shape[0])
]
non_strong_pairs = [
    (DNQWrapper(_UPGrad2()), matrix) for matrix in non_strong_matrices if is_ok(matrix.shape[0])
]
requires_grad_pairs = [(DNQWrapper(_UPGrad2()), torch.ones(4, 5, requires_grad=True))]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: DNQWrapper, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_non_conflicting(aggregator: DNQWrapper, matrix: Tensor):
    assert_non_conflicting(aggregator, matrix, atol=3e-04, rtol=3e-04)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_linear_under_scaling(aggregator: DNQWrapper, matrix: Tensor):
    assert_linear_under_scaling(aggregator, matrix, n_runs=5, atol=3e-02, rtol=3e-02)


@mark.parametrize(["aggregator", "matrix"], non_strong_pairs)
def test_strongly_stationary(aggregator: DNQWrapper, matrix: Tensor):
    assert_strongly_stationary(aggregator, matrix, threshold=5e-03)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: DNQWrapper, matrix: Tensor):
    assert_non_differentiable(aggregator, matrix)


def test_representations():
    pass  # TODO
