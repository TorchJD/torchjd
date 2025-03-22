from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch import Tensor
from unit._utils import ExceptionContext

from torchjd.aggregation import Krum

from ._inputs import scaled_matrices_2_plus_rows, typical_matrices_2_plus_rows
from ._property_testers import ExpectedStructureProperty


@mark.parametrize("aggregator", [Krum(n_byzantine=1)])
class TestKrum(ExpectedStructureProperty):
    # Override the parametrization of some property-testing methods because Krum only works on
    # matrices with >= 2 rows.
    @classmethod
    @mark.parametrize("matrix", scaled_matrices_2_plus_rows + typical_matrices_2_plus_rows)
    def test_expected_structure_property(cls, aggregator: Krum, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)


@mark.parametrize(
    ["n_byzantine", "expectation"],
    [
        (-5, raises(ValueError)),
        (-1, raises(ValueError)),
        (0, does_not_raise()),
        (1, does_not_raise()),
        (5, does_not_raise()),
    ],
)
def test_n_byzantine_check(n_byzantine: int, expectation: ExceptionContext):
    with expectation:
        _ = Krum(n_byzantine=n_byzantine, n_selected=1)


@mark.parametrize(
    ["n_selected", "expectation"],
    [
        (-5, raises(ValueError)),
        (-1, raises(ValueError)),
        (0, raises(ValueError)),
        (1, does_not_raise()),
        (5, does_not_raise()),
    ],
)
def test_n_selected_check(n_selected: int, expectation: ExceptionContext):
    with expectation:
        _ = Krum(n_byzantine=1, n_selected=n_selected)


@mark.parametrize(
    ["n_byzantine", "n_selected", "n_rows", "expectation"],
    [
        (1, 1, 3, raises(ValueError)),
        (1, 1, 4, does_not_raise()),
        (1, 4, 4, does_not_raise()),
        (12, 4, 14, raises(ValueError)),
        (12, 4, 15, does_not_raise()),
        (12, 15, 15, does_not_raise()),
        (12, 16, 15, raises(ValueError)),
    ],
)
def test_matrix_shape_check(
    n_byzantine: int, n_selected: int, n_rows: int, expectation: ExceptionContext
):
    aggregator = Krum(n_byzantine=n_byzantine, n_selected=n_selected)
    matrix = torch.ones([n_rows, 5])

    with expectation:
        _ = aggregator(matrix)


def test_representations():
    A = Krum(n_byzantine=1, n_selected=2)
    assert repr(A) == "Krum(n_byzantine=1, n_selected=2)"
    assert str(A) == "Krum1-2"
