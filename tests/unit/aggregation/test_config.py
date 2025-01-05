from pytest import mark

from torchjd.aggregation import ConFIG

from ._property_testers import ExpectedStructureProperty, PermutationInvarianceProperty


@mark.parametrize("aggregator", [ConFIG()])
class TestConFIGLeastSquares(ExpectedStructureProperty, PermutationInvarianceProperty):
    pass


@mark.parametrize("aggregator", [ConFIG(use_least_square=False)])
class TestConFIGPseudoInverse(ExpectedStructureProperty):
    # For some reason, one of the output values is infinite when using the pseudo-inverse, making
    # the permutation-invariance test fail.
    pass


def test_representations():
    A = ConFIG(use_least_square=True)
    assert repr(A) == "ConFIG(use_least_square=True)"
    assert str(A) == "ConFIG"
