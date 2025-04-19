import torch
from pytest import mark
from torch import Tensor
from torch.testing import assert_close

from torchjd.aggregation import NashMTL

from ._inputs import nash_mtl_matrices
from ._property_testers import ExpectedStructureProperty


def _make_aggregator(matrix: Tensor) -> NashMTL:
    return NashMTL(n_tasks=matrix.shape[0])


_aggregators = [_make_aggregator(matrix) for matrix in nash_mtl_matrices]


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
    @mark.parametrize(["aggregator", "matrix"], zip(_aggregators, nash_mtl_matrices))
    def test_expected_structure_property(cls, aggregator: NashMTL, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)


@mark.filterwarnings("ignore: You are solving a parameterized problem that is not DPP.")
def test_nash_mtl_reset():
    """
    Tests that the reset method of NashMTL correctly resets its internal state, by verifying that
    the result is the same after reset as it is right after instantiation.

    To ensure that the aggregations are not all the same, we create different matrices to aggregate.
    """

    matrices = [torch.randn(3, 5) for _ in range(4)]
    aggregator = NashMTL(n_tasks=3, update_weights_every=3)
    expecteds = [aggregator(matrix) for matrix in matrices]

    aggregator.reset()
    results = [aggregator(matrix) for matrix in matrices]

    for result, expected in zip(results, expecteds):
        assert_close(result, expected)


def test_representations():
    A = NashMTL(n_tasks=2)
    assert repr(A) == "NashMTL(n_tasks=2)"
    assert str(A) == "NashMTL"
