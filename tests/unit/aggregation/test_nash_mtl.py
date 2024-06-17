from torchjd.aggregation import NashMTL


def test_representations():
    A = NashMTL(n_tasks=2)
    assert repr(A) == "NashMTL(n_tasks=2)"
    assert str(A) == "NashMTL"
