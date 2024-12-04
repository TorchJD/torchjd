from pytest import mark
from torch import Tensor

from torchjd.aggregation import Krum

from ._inputs import scaled_matrices_2_plus_rows
from ._property_testers import ExpectedStructureProperty


@mark.parametrize("aggregator", [Krum(n_byzantine=1)])
class TestKrum(ExpectedStructureProperty):
    # Override the parametrization of some property-testing methods because Krum only works on
    # matrices with >= 2 rows.
    @classmethod
    @mark.parametrize("matrix", scaled_matrices_2_plus_rows)
    def test_expected_structure_property(cls, aggregator: Krum, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)


def test_representations():
    A = Krum(n_byzantine=1, n_selected=2)
    assert repr(A) == "Krum(n_byzantine=1, n_selected=2)"
    assert str(A) == "Krum1-2"
