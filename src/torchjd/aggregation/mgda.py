import torch
from torch import Tensor

from .bases import PSDMatrix, _GramianWeightedAggregator, _Weighting


class MGDA(_GramianWeightedAggregator):
    r"""
    :class:`~torchjd.aggregation.bases.Aggregator` performing the gradient aggregation
    step of `Multiple-gradient descent algorithm (MGDA) for multiobjective optimization
    <https://www.sciencedirect.com/science/article/pii/S1631073X12000738>`_. The implementation is
    based on Algorithm 2 of `Multi-Task Learning as Multi-Objective Optimization
    <https://proceedings.neurips.cc/paper_files/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf>`_.

    :param epsilon: The value of :math:`\hat{\gamma}` below which we stop the optimization.
    :param max_iters: The maximum number of iterations of the optimization loop.

    .. admonition::
        Example

        Use MGDA to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import MGDA
        >>>
        >>> A = MGDA()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([1.1921e-07, 1.0000e+00, 1.0000e+00])
    """

    def __init__(self, epsilon: float = 0.001, max_iters: int = 100):
        super().__init__(_MGDAWeighting(epsilon=epsilon, max_iters=max_iters))
        self._max_iters = max_iters

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(epsilon={self.weighting.weighting.epsilon}, max_iters="
            f"{self._max_iters})"
        )


class _MGDAWeighting(_Weighting[PSDMatrix]):
    r"""
    :class:`~torchjd.aggregation.bases._Weighting` that extracts weights using Algorithm
    2 of `Multi-Task Learning as Multi-Objective Optimization
    <https://proceedings.neurips.cc/paper_files/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf>`_.

    :param epsilon: The value of :math:`\hat{\gamma}` below which we stop the optimization.
    :param max_iters: The maximum number of iterations of the optimization loop.
    """

    def __init__(self, epsilon: float, max_iters: int):
        super().__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters

    def forward(self, gramian: Tensor) -> Tensor:
        """
        This is the Frank-Wolfe solver in Algorithm 2 of `Multi-Task Learning as Multi-Objective
        Optimization
        <https://proceedings.neurips.cc/paper_files/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf>`_.
        """
        device = gramian.device
        dtype = gramian.dtype

        alpha = torch.ones(gramian.shape[0], device=device, dtype=dtype) / gramian.shape[0]
        for i in range(self.max_iters):
            t = torch.argmin(gramian @ alpha)
            e_t = torch.zeros(gramian.shape[0], device=device, dtype=dtype)
            e_t[t] = 1.0
            a = alpha @ (gramian @ e_t)
            b = alpha @ (gramian @ alpha)
            c = e_t @ (gramian @ e_t)
            if c <= a:
                gamma = 1.0
            elif b <= a:
                gamma = 0.0
            else:
                gamma = (b - a) / (b + c - 2 * a)
            alpha = (1 - gamma) * alpha + gamma * e_t
            if gamma < self.epsilon:
                break
        return alpha
