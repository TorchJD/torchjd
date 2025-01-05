from pytest import mark

from torchjd.aggregation import ConFIG

from ._property_testers import ExpectedStructureProperty


# For some reason, some permutation-invariance property tests fail when use_least_square=False or
# on Windows.
@mark.parametrize("aggregator", [ConFIG(), ConFIG(use_least_square=False)])
class TestConFIGLeastSquares(ExpectedStructureProperty):
    pass


def test_representations():
    A = ConFIG(use_least_square=True)
    assert repr(A) == "ConFIG(pref_vector=None, use_least_square=True)"
    assert str(A) == "ConFIG"
