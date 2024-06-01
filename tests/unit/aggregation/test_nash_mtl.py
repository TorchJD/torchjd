from torchjd.aggregation import NashMTLWeighting, WeightedAggregator


def test_representations():
    weighting = NashMTLWeighting(n_tasks=2)
    assert repr(weighting) == "NashMTLWeighting()"
    assert str(weighting) == "NashMTLWeighting"

    aggregator = WeightedAggregator(weighting)
    assert repr(aggregator) == "WeightedAggregator(weighting=NashMTLWeighting())"
    assert str(aggregator) == "NashMTL"
