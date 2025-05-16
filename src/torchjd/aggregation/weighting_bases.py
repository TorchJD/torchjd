from abc import ABC, abstractmethod

from torch import Tensor, nn


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
