from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from torchjd.aggregation._utils.gramian import compute_gramian


class _Weighting(nn.Module, ABC):
    r"""
    Abstract base class for all weighting methods. It has the role of extracting a vector of weights
    of dimension :math:`m` from a matrix of dimension :math:`m \times n`.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, matrix: Tensor) -> Tensor:
        """Computes the vector of weights from the input matrix."""

    # Override to make type hints and documentation more specific
    def __call__(self, matrix: Tensor) -> Tensor:
        """Computes the vector of weights from the input matrix and applies all registered hooks."""

        return super().__call__(matrix)


class _GramianBasedWeighting(_Weighting, ABC):
    """
    Abstract base class for all weighting methods that only rely on the matrix through its Gramian.
    """

    def forward(self, matrix: Tensor) -> Tensor:
        gramian = compute_gramian(matrix)
        return self.weights_from_gramian(gramian)

    @abstractmethod
    def weights_from_gramian(self, gramian: Tensor) -> Tensor:
        """Computes the vector of weights from a gramian matrix."""


class _RowDimensionBasedWeighting(_GramianBasedWeighting, ABC):
    """
    Abstract base class for all weighting methods that only rely on the matrix through its row
    dimension.
    """

    def weights_from_gramian(self, gramian: Tensor) -> Tensor:
        return self.weights_from_dimension(*_extract_row_dimension_and_metadata(gramian))

    # Override forward to avoid computing the Gramian
    def forward(self, matrix: Tensor) -> Tensor:
        return self.weights_from_dimension(*_extract_row_dimension_and_metadata(matrix))

    @abstractmethod
    def weights_from_dimension(self, m: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Computes the vector of weights from the row dimension of a matrix."""


def _extract_row_dimension_and_metadata(matrix):
    m = matrix.shape[0]
    device = matrix.device
    dtype = matrix.dtype
    return m, device, dtype
