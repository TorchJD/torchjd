import torch
from torch import Tensor
from torch.nn import functional as F

from .bases import _WeightedAggregator, _Weighting


class Krum(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` for adversarial federated learning, as defined
    in `Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
    <https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf>`_.

    :param n_byzantine: The number of rows of the input matrix that can come from an adversarial
        source.
    :param n_selected: The number of selected rows in the context of Multi-Krum. Defaults to 1.

    .. admonition::
        Example

        Use Multi-Krum to aggregate a matrix with 1 adversarial row.

        >>> from torch import tensor
        >>> from torchjd.aggregation import Krum
        >>>
        >>> A = Krum(n_byzantine=1, n_selected=4)
        >>> J = tensor([
        ...     [1.,     1., 1.],
        ...     [1.,     0., 1.],
        ...     [75., -666., 23],  # adversarial row
        ...     [1.,     2., 3.],
        ...     [2.,     0., 1.],
        ... ])
        >>>
        >>> A(J)
        tensor([1.2500, 0.7500, 1.5000])
    """

    def __init__(self, n_byzantine: int, n_selected: int = 1):
        super().__init__(weighting=_KrumWeighting(n_byzantine=n_byzantine, n_selected=n_selected))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_byzantine={self.weighting.n_byzantine}, n_selected="
            f"{self.weighting.n_selected})"
        )

    def __str__(self) -> str:
        return f"Krum{self.weighting.n_byzantine}-{self.weighting.n_selected}"


class _KrumWeighting(_Weighting):
    """
    :class:`~torchjd.aggregation.bases._Weighting` that extracts weights using the
    (Multi-)Krum aggregation rule, as defined in `Machine Learning with Adversaries: Byzantine
    Tolerant Gradient Descent
    <https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf>`_.

    :param n_byzantine: The number of rows of the input matrix that can come from an adversarial
        source.
    :param n_selected: The number of selected rows in the context of Multi-Krum. Defaults to 1.
    """

    def __init__(self, n_byzantine: int, n_selected: int):
        super().__init__()
        if n_byzantine < 0:
            raise ValueError(
                "Parameter `n_byzantine` should be a non-negative integer. Found `n_byzantine = "
                f"{n_byzantine}`."
            )

        if n_selected < 1:
            raise ValueError(
                "Parameter `n_selected` should be a positive integer. Found `n_selected = "
                f"{n_selected}`."
            )

        self.n_byzantine = n_byzantine
        self.n_selected = n_selected

    def forward(self, matrix: Tensor) -> Tensor:
        self._check_matrix_shape(matrix)

        distances = torch.cdist(matrix, matrix, compute_mode="donot_use_mm_for_euclid_dist")
        n_closest = matrix.shape[0] - self.n_byzantine - 2
        smallest_distances, _ = torch.topk(distances, k=n_closest + 1, largest=False)
        smallest_distances_excluding_self = smallest_distances[:, 1:]
        scores = smallest_distances_excluding_self.sum(dim=1)

        _, selected_indices = torch.topk(scores, k=self.n_selected, largest=False)
        one_hot_selected_indices = F.one_hot(selected_indices, num_classes=matrix.shape[0])
        weights = one_hot_selected_indices.sum(dim=0).to(dtype=matrix.dtype) / self.n_selected

        return weights

    def _check_matrix_shape(self, matrix: Tensor) -> None:
        min_rows = self.n_byzantine + 3
        if matrix.shape[0] < min_rows:
            raise ValueError(
                f"Parameter `matrix` should have at least {min_rows} rows (n_byzantine + 3). Found "
                f"`matrix` with {matrix.shape[0]} rows."
            )

        if matrix.shape[0] < self.n_selected:
            raise ValueError(
                f"Parameter `matrix` should have at least {self.n_selected} rows (n_selected). "
                f"Found `matrix` with {matrix.shape[0]} rows."
            )
