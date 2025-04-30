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
    :param max_iter: The maximal number of iteration of the solver.
    :param eps: The convergence threshold of the solver.

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
        tensor([0.2924, 1.9006, 1.9006])
    """

    def __init__(self, pref_vector: Tensor | None = None, max_iter: int = 200, eps: float = 1e-07):
        weighting = pref_vector_to_weighting(pref_vector, default=_MeanWeighting())
        self._pref_vector = pref_vector

        super().__init__(weighting=_UPGradWrapper(weighting, max_iter, eps))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, max_iter="
            f"{self.weighting.max_iter}, eps={self.weighting.eps})"
        )

    def __str__(self) -> str:
        return f"UPGrad{pref_vector_to_str_suffix(self._pref_vector)}"


class _UPGradWrapper(_Weighting):
    """
    Wrapper of :class:`~torchjd.aggregation.bases._Weighting` that changes the weights vector such
    that each weighted row is projected onto the dual cone of all rows.

    :param weighting: The wrapped weighting.
    :param max_iter: The maximal number of iteration of the solver.
    :param eps: The convergence threshold of the solver.
    """

    def __init__(self, weighting: _Weighting, max_iter: int, eps: float):
        super().__init__()
        self.weighting = weighting
        self.max_iter = max_iter
        self.eps = eps

    def forward(self, matrix: Tensor) -> Tensor:
        U = torch.diag(self.weighting(matrix))
        G = compute_gramian(matrix)
        W = project_weights(U, G, self.max_iter, self.eps)
        return torch.sum(W, dim=0)
