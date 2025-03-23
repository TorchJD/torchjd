from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch import Tensor
from unit._utils import ExceptionContext

from torchjd.aggregation import Aggregator, TrimmedMean

from ._inputs import scaled_matrices_2_plus_rows, typical_matrices_2_plus_rows
from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@mark.parametrize("aggregator", [TrimmedMean(trim_number=1)])
class TestTrimmedMean(ExpectedStructureProperty, PermutationInvarianceProperty):
    # Override the parametrization of some property-testing methods because `TrimmedMean` with
    # `trim_number=1` only works on matrices with >= 2 rows.
    @classmethod
    @mark.parametrize("matrix", scaled_matrices_2_plus_rows + typical_matrices_2_plus_rows)
    def test_expected_structure_property(cls, aggregator: TrimmedMean, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)

    @classmethod
    @mark.parametrize("matrix", typical_matrices_2_plus_rows)
    def test_permutation_invariance_property(cls, aggregator: Aggregator, matrix: Tensor):
        cls._assert_permutation_invariance_property(aggregator, matrix)


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


def test_representations():
    aggregator = TrimmedMean(trim_number=2)
    assert repr(aggregator) == "TrimmedMean(trim_number=2)"
    assert str(aggregator) == "TM2"
