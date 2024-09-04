import torch
from torch import Tensor

from .bases import Aggregator


class TrimmedMean(Aggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` for adversarial federated learning, that trims
    the most extreme values of the input matrix, before averaging its rows, as defined in
    `Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates
    <https://proceedings.mlr.press/v80/yin18a/yin18a.pdf>`_.

    :param trim_number: The number of maximum and minimum values to remove from each column of the
        input matrix (note that ``2 * trim_number`` values are removed from each column).

    .. admonition::
        Example

        Remove the maximum and the minimum value from each column of the ``matrix``, then average
        the rows of the remaining matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import TrimmedMean
        >>>
        >>> A = TrimmedMean(trim_number=1)
        >>> J = tensor([
        ...     [ 1e11,     3],
        ...     [    1, -1e11],
        ...     [-1e10,  1e10],
        ...     [    2,     2],
        ... ])
        >>>
        >>> A(J)
        tensor([1.5000, 2.5000])
    """

    def __init__(self, trim_number: int):
        super().__init__()
        if trim_number < 0:
            raise ValueError(
                "Parameter `trim_number` should be a non-negative integer. Found `trim_number` = "
                f"{trim_number}`."
            )
        self.trim_number = trim_number

    def forward(self, matrix: Tensor) -> Tensor:
        self._check_is_matrix(matrix)
        self._check_matrix_has_enough_rows(matrix)
        self._check_is_finite(matrix)

        n_rows = matrix.shape[0]
        n_remaining = n_rows - 2 * self.trim_number

        sorted_matrix, _ = torch.sort(matrix, dim=0)
        trimmed = torch.narrow(sorted_matrix, dim=0, start=self.trim_number, length=n_remaining)
        vector = trimmed.mean(dim=0)
        return vector

    def _check_matrix_has_enough_rows(self, matrix: Tensor) -> None:
        min_rows = 1 + 2 * self.trim_number
        n_rows = matrix.shape[0]
        if n_rows < min_rows:
            raise ValueError(
                f"Parameter `matrix` should be a matrix of at least {min_rows} rows "
                f"(i.e. `2 * trim_number + 1`). Found `matrix` of shape `{matrix.shape}`."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(trim_number={self.trim_number})"

    def __str__(self) -> str:
        return f"TM{self.trim_number}"
