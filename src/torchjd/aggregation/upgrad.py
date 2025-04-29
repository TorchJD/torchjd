import torch
from torch import Tensor

from ._dual_cone_utils import project_weights
from ._gramian_utils import compute_gramian
from ._pref_vector_utils import pref_vector_to_str_suffix, pref_vector_to_weighting
from .bases import _WeightedAggregator, _Weighting
from .mean import _MeanWeighting


class UPGrad(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that projects each row of the input matrix onto
    the dual cone of all rows of this matrix, and that combines the result, as proposed in
    `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param pref_vector: The preference vector used to combine the projected rows. If not provided,
        defaults to the simple averaging of the projected rows.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param solver: The solver used to optimize the underlying optimization problem.

    .. admonition::
        Example

        Use UPGrad to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import UPGrad
        >>>
        >>> A = UPGrad()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.2929, 1.9004, 1.9004])
    """

    def __init__(
        self,
        pref_vector: Tensor | None = None,
        norm_eps: float = 0.0001,
    ):
        weighting = pref_vector_to_weighting(pref_vector, default=_MeanWeighting())
        self._pref_vector = pref_vector

        super().__init__(weighting=_UPGradWrapper(weighting=weighting, norm_eps=norm_eps))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, norm_eps="
            f"{self.weighting.norm_eps})"
        )

    def __str__(self) -> str:
        return f"UPGrad{pref_vector_to_str_suffix(self._pref_vector)}"


class _UPGradWrapper(_Weighting):
    """
    Wrapper of :class:`~torchjd.aggregation.bases._Weighting` that changes the weights vector such
    that each weighted row is projected onto the dual cone of all rows.

    :param weighting: The wrapped weighting.
    :param norm_eps: A small value to avoid division by zero when normalizing.
    :param solver: The solver used to optimize the underlying optimization problem.
    """

    def __init__(
        self,
        weighting: _Weighting,
        norm_eps: float,
    ):
        super().__init__()
        self.weighting = weighting
        self.norm_eps = norm_eps

    def forward(self, matrix: Tensor) -> Tensor:
        U = torch.diag(self.weighting(matrix))
        G = compute_gramian(matrix)
        W = project_weights(U, G)
        return torch.sum(W, dim=0)
