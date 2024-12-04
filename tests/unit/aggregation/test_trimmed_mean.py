from pytest import mark
from torch import Tensor

from torchjd.aggregation import Aggregator, TrimmedMean

from ._inputs import matrices_2_plus_rows, scaled_matrices_2_plus_rows
from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@mark.parametrize("aggregator", [TrimmedMean(trim_number=1)])
class TestTrimmedMean(ExpectedStructureProperty, PermutationInvarianceProperty):
    # Override the parametrization of some property-testing methods because `TrimmedMean` with
    # `trim_number=1` only works on matrices with >= 2 rows.
    @classmethod
    @mark.parametrize("matrix", scaled_matrices_2_plus_rows)
    def test_expected_structure_property(cls, aggregator: TrimmedMean, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)

    @classmethod
    @mark.parametrize("matrix", matrices_2_plus_rows)
    def test_permutation_invariance_property(cls, aggregator: Aggregator, matrix: Tensor):
        cls._assert_permutation_invariance_property(aggregator, matrix)


def test_representations():
    aggregator = TrimmedMean(trim_number=2)
    assert repr(aggregator) == "TrimmedMean(trim_number=2)"
    assert str(aggregator) == "TM2"
