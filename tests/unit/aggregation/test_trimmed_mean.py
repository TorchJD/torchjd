from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from unit._utils import ExceptionContext

from torchjd.aggregation import TrimmedMean

from ._asserts import assert_expected_structure, assert_permutation_invariant
from ._inputs import scaled_matrices_2_plus_rows, typical_matrices_2_plus_rows

scaled_pairs = [(TrimmedMean(trim_number=1), matrix) for matrix in scaled_matrices_2_plus_rows]
typical_pairs = [(TrimmedMean(trim_number=1), matrix) for matrix in typical_matrices_2_plus_rows]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: TrimmedMean, matrix: Tensor):
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], typical_pairs)
def test_permutation_invariant(aggregator: TrimmedMean, matrix: Tensor):
    assert_permutation_invariant(aggregator, matrix)


@mark.parametrize(
    ["trim_number", "expectation"],
    [
        (-5, raises(ValueError)),
        (-1, raises(ValueError)),
        (0, does_not_raise()),
        (1, does_not_raise()),
        (5, does_not_raise()),
    ],
)
def test_trim_number_check(trim_number: int, expectation: ExceptionContext):
    with expectation:
        _ = TrimmedMean(trim_number=trim_number)


@mark.parametrize(
    ["n_rows", "trim_number", "expectation"],
    [
        (1, 0, does_not_raise()),
        (1, 1, raises(ValueError)),
        (10, 0, does_not_raise()),
        (10, 4, does_not_raise()),
        (10, 5, raises(ValueError)),
    ],
)
def test_matrix_shape_check(n_rows: int, trim_number: int, expectation: ExceptionContext):
    matrix = torch.ones([n_rows, 5])
    aggregator = TrimmedMean(trim_number=trim_number)

    with expectation:
        _ = aggregator(matrix)


def test_one_nan():
    aggregator = TrimmedMean(trim_number=1)
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.nan
    result = aggregator(matrix)
    assert_close(result, torch.ones_like(result))


def test_full_nan():
    aggregator = TrimmedMean(trim_number=1)
    matrix = torch.full([10, 100], torch.nan)
    result = aggregator(matrix)
    assert result.isnan().all()


def test_one_inf():
    aggregator = TrimmedMean(trim_number=1)
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = torch.inf
    result = aggregator(matrix)
    assert_close(result, torch.ones_like(result))


def test_full_inf():
    aggregator = TrimmedMean(trim_number=1)
    matrix = torch.full([10, 100], torch.inf)
    result = aggregator(matrix)
    assert result.eq(torch.full_like(result, torch.inf)).all()


def test_one_neg_inf():
    aggregator = TrimmedMean(trim_number=1)
    matrix = torch.full([10, 100], 1.0)
    matrix[0, 0] = -torch.inf
    result = aggregator(matrix)
    assert_close(result, torch.ones_like(result))


def test_full_neg_inf():
    aggregator = TrimmedMean(trim_number=1)
    matrix = torch.full([10, 100], -torch.inf)
    result = aggregator(matrix)
    assert result.eq(torch.full_like(result, -torch.inf)).all()


def test_representations():
    aggregator = TrimmedMean(trim_number=2)
    assert repr(aggregator) == "TrimmedMean(trim_number=2)"
    assert str(aggregator) == "TM2"
