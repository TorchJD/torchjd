import pytest
from unit.aggregation.utils.property_testers import ExpectedShapeProperty

from torchjd.aggregation._normalizing import _NormalizingWrapper
from torchjd.aggregation.bases import _WeightedAggregator
from torchjd.aggregation.mean import _MeanWeighting
from torchjd.aggregation.mgda import _MGDAWeighting
from torchjd.aggregation.sum import _SumWeighting


@pytest.mark.parametrize(
    "aggregator",
    [
        _WeightedAggregator(_NormalizingWrapper(_SumWeighting(), norm_p=0.5, norm_value=1.0)),
        _WeightedAggregator(_NormalizingWrapper(_SumWeighting(), norm_p=1.0, norm_value=1.0)),
        _WeightedAggregator(_NormalizingWrapper(_MeanWeighting(), norm_p=2.0, norm_value=1.0)),
        _WeightedAggregator(
            _NormalizingWrapper(
                _MGDAWeighting(epsilon=0.001, max_iters=100), norm_p=10.0, norm_value=1.0
            )
        ),
    ],
)
class TestNormalizing(ExpectedShapeProperty):
    pass
