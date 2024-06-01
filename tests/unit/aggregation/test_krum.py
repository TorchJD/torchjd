import pytest
from torch import Tensor
from unit.aggregation.utils.inputs import scaled_matrices_2_plus_rows
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation import KrumWeighting, WeightedAggregator


@pytest.mark.parametrize("aggregator", [WeightedAggregator(KrumWeighting(n_byzantine=1))])
class TestKrum(ExpectedShapeProperty):
    # Override the parametrization of some property-testing methods because Krum only works on
    # matrices with >= 2 rows.
    @classmethod
    @pytest.mark.parametrize("matrix", scaled_matrices_2_plus_rows)
    def test_expected_shape_property(cls, aggregator: WeightedAggregator, matrix: Tensor):
        cls._assert_expected_shape_property(aggregator, matrix)


def test_representations():
    weighting = KrumWeighting(n_byzantine=1, n_selected=2)
    assert repr(weighting) == "KrumWeighting(n_byzantine=1, n_selected=2)"
    assert str(weighting) == "Krum1-2Weighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == (
        "WeightedAggregator(weighting=KrumWeighting(n_byzantine=1, n_selected=2))"
    )
    assert str(aggregator) == "Krum1-2"
