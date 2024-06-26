import pytest
from torch import Tensor
from torch.testing import assert_close
from unit.aggregation.utils import (
    ExpectedShapeProperty,
    NonConflictingProperty,
    matrices,
    stationary_matrices,
)

from torchjd.aggregation import CAGrad, Mean


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated:")
@pytest.mark.parametrize("aggregator", [CAGrad(c=0.5)])
class TestCAGrad(ExpectedShapeProperty):
    pass


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated:")
@pytest.mark.parametrize("aggregator", [CAGrad(c=1.0), CAGrad(c=2.0)])
class TestCAGradNonConflicting(NonConflictingProperty):
    """Tests that CAGrad is non-conflicting when c >= 1 (it should not hold when c < 1)"""

    pass


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated:")
@pytest.mark.parametrize("matrix", stationary_matrices + matrices)
def test_equivalence_mean(matrix: Tensor):
    """Tests that CAGrad is equivalent to Mean when c=0."""

    ca_grad = CAGrad(c=0.0)
    mean = Mean()

    result = ca_grad(matrix)
    expected = mean(matrix)

    assert_close(result, expected)


def test_representations():
    A = CAGrad(c=0.5, norm_eps=0.0001)
    assert repr(A) == "CAGrad(c=0.5, norm_eps=0.0001)"
    assert str(A) == "CAGrad0.5"
