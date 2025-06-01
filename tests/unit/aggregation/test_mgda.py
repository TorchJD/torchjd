import torch
from pytest import mark
from torch import Tensor
from torch.testing import assert_close

from torchjd.aggregation import MGDA
from torchjd.aggregation._mgda import _MGDAWeighting
from torchjd.aggregation._utils.gramian import compute_gramian

from ._asserts import (
    assert_expected_structure,
    assert_non_conflicting,
    assert_permutation_invariant,
)
from ._inputs import scaled_matrices, typical_matrices

scaled_pairs = [(MGDA(), matrix) for matrix in scaled_matrices]
typical_pairs = [(MGDA(), matrix) for matrix in typical_matrices]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: MGDA, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_non_conflicting(aggregator: MGDA, matrix: Tensor):
    assert_non_conflicting(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: MGDA, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(
    "shape",
    [
        (5, 7),
        (9, 37),
        (2, 14),
        (32, 114),
        (50, 100),
    ],
)
def test_mgda_satisfies_kkt_conditions(shape: tuple[int, int]):
    matrix = torch.randn(shape)
    gramian = compute_gramian(matrix)

    weighting = _MGDAWeighting(epsilon=1e-05, max_iters=1000)
    weights = weighting(gramian)

    output_direction = gramian @ weights  # Stationarity
    lamb = -weights @ output_direction  # Complementary slackness
    mu = output_direction + lamb

    # Primal feasibility
    positive_weights = weights[weights >= 0]
    assert_close(positive_weights.norm(), weights.norm())

    weights_sum = weights.sum()
    assert_close(weights_sum, torch.ones([]))

    # Dual feasibility
    positive_mu = mu[mu >= 0]
    assert_close(positive_mu.norm(), mu.norm(), atol=1e-02, rtol=0.0)


def test_one_nan():
    aggregator = MGDA()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.nan
    result = aggregator(matrix)
    assert result.isnan().all()


def test_full_nan():
    aggregator = MGDA()
    matrix = torch.full([10, 100], torch.nan)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_inf():
    aggregator = MGDA()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.inf
    result = aggregator(matrix)
    assert result.isnan().all()


def test_full_inf():
    aggregator = MGDA()
    matrix = torch.full([10, 100], torch.inf)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_neg_inf():
    aggregator = MGDA()
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = -torch.inf
    result = aggregator(matrix)
    assert result.isnan().all()


def test_full_neg_inf():
    aggregator = MGDA()
    matrix = torch.full([10, 100], -torch.inf)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_representations():
    A = MGDA(epsilon=0.001, max_iters=100)
    assert repr(A) == "MGDA(epsilon=0.001, max_iters=100)"
    assert str(A) == "MGDA"
