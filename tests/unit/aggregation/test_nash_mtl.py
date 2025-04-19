from pytest import mark
from torch import Tensor

from torchjd.aggregation import NashMTL

from ._inputs import nashmtl_matrices
from ._property_testers import ExpectedStructureProperty


def _make_aggregator(matrix: Tensor) -> NashMTL:
    return NashMTL(n_tasks=matrix.shape[0])


_aggregators = [_make_aggregator(matrix) for matrix in nashmtl_matrices]


@mark.filterwarnings(
    "ignore:Solution may be inaccurate.",
    "ignore:You are solving a parameterized problem that is not DPP.",
)
class TestNashMTL(ExpectedStructureProperty):
    # Override the parametrization of `test_expected_structure_property` to make the test use the
    # right aggregator with each matrix.

    # Note that as opposed to most aggregators, the ExpectedStructureProperty is only tested with
    # non-scaled matrices, and with matrices of > 1 row. Otherwise, NashMTL fails.
    @classmethod
    @mark.parametrize(["aggregator", "matrix"], zip(_aggregators, nashmtl_matrices))
    def test_expected_structure_property(cls, aggregator: NashMTL, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)


def test_representations():
    A = NashMTL(n_tasks=2)
    assert repr(A) == "NashMTL(n_tasks=2)"
    assert str(A) == "NashMTL"
