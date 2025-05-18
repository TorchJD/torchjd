import torch
from torch import Tensor
from torch.testing import assert_close

from torchjd.aggregation.weighting_bases import _RowDimensionBasedWeighting


class FakeRowDimensionBasedWeighting(_RowDimensionBasedWeighting):
    def weights_from_dimension(self, m: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        weights = torch.ones(m, device=device, dtype=dtype)
        return weights


def test_equivalence_weights_from_gramian_or_dimension():
    weighting = FakeRowDimensionBasedWeighting()

    matrix = torch.rand([4, 7])
    gramian = matrix.T @ matrix

    weights_from_gramian = weighting.weights_from_gramian(gramian)
    weights_from_dimension = weighting.weights_from_dimension(
        gramian.shape[0], gramian.device, gramian.dtype
    )

    assert_close(weights_from_dimension, weights_from_gramian)
