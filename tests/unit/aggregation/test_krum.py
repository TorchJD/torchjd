import pytest
from torch import Tensor

from torchjd.aggregation import Krum

from .utils import ExpectedShapeProperty, scaled_matrices_2_plus_rows


@pytest.mark.parametrize("aggregator", [Krum(n_byzantine=1)])
class TestKrum(ExpectedShapeProperty):
    # Override the parametrization of some property-testing methods because Krum only works on
    # matrices with >= 2 rows.
    @classmethod
    @pytest.mark.parametrize("matrix", scaled_matrices_2_plus_rows)
    def test_expected_shape_property(cls, aggregator: Krum, matrix: Tensor):
        cls._assert_expected_shape_property(aggregator, matrix)


def test_representations():
    A = Krum(n_byzantine=1, n_selected=2)
    assert repr(A) == "Krum(n_byzantine=1, n_selected=2)"
    assert str(A) == "Krum1-2"
