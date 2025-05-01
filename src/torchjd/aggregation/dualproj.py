from torch import Tensor

from ._dual_cone_utils import project_weights
from ._gramian_utils import compute_gramian
from ._non_differentiable import raise_non_differentiable_error
from ._pref_vector_utils import pref_vector_to_str_suffix, pref_vector_to_weighting
from .bases import _WeightedAggregator, _Weighting
from .mean import _MeanWeighting


class DualProj(_WeightedAggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` that averages the rows of the input matrix, and
    projects the result onto the dual cone of the rows of the matrix. This corresponds to the
    solution to Equation 11 of `Gradient Episodic Memory for Continual Learning
    <https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf>`_.

    :param pref_vector: The preference vector used to combine the rows. If not provided, defaults to
        the simple averaging.
    :param max_iter: The maximal number of iterations of the solver.
    :param eps: The convergence threshold of the solver. A lower value leads to a higher precision
        but a potentially larger number of iterations.

    .. admonition::
        Example

        Use DualProj to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import DualProj
        >>>
        >>> A = DualProj()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.5556, 1.1111, 1.1111])
    """

    def __init__(self, pref_vector: Tensor | None = None, max_iter: int = 100, eps: float = 1e-05):
        weighting = pref_vector_to_weighting(pref_vector, default=_MeanWeighting())
        self._pref_vector = pref_vector

        super().__init__(weighting=_DualProjWrapper(weighting, max_iter, eps))

        # This prevents considering the computed weights as constant w.r.t. the matrix.
        self.register_full_backward_pre_hook(raise_non_differentiable_error)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)}, max_iter="
            f"{self.weighting.max_iter}, eps={self.weighting.eps})"
        )

    def __str__(self) -> str:
        return f"DualProj{pref_vector_to_str_suffix(self._pref_vector)}"


class _DualProjWrapper(_Weighting):
    """
    Wrapper of :class:`~torchjd.aggregation.bases._Weighting` that changes the extracted
    weight vector such the corresponding aggregation is projected onto the dual cone of the rows
    of the input matrix. This corresponds to the solution to Equation 11 of `Gradient Episodic
    Memory for Continual Learning
    <https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf>`_.

    :param weighting: The wrapped :class:`~torchjd.aggregation.bases._Weighting`
        responsible for extracting weight vectors from the input matrices.
    :param max_iter: The maximal number of iteration of the solver.
    :param eps: The convergence threshold of the solver.
    """

    def __init__(self, weighting: _Weighting, max_iter: int, eps: float):
        super().__init__()
        self.weighting = weighting
        self.max_iter = max_iter
        self.eps = eps

    def forward(self, matrix: Tensor) -> Tensor:
        u = self.weighting(matrix)
        G = compute_gramian(matrix)
        w = project_weights(u, G, self.max_iter, self.eps)
        return w
