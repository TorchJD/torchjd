# The code of this file was partly adapted from
# https://github.com/tum-pbs/ConFIG/tree/main/conflictfree.
# It is therefore also subject to the following license.
#
# MIT License
#
# Copyright (c) 2024 TUM Physics-based Simulation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import Tensor

from torchjd.aggregation.bases import Aggregator


class ConFIG(Aggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` as defined in Equation 2 of `ConFIG: Towards
    Conflict-free Training of Physics Informed Neural Networks <https://arxiv.org/pdf/2408.11104>`_.

    .. admonition::
        Example

        Use ConFIG to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import ConFIG
        >>>
        >>> A = ConFIG()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        tensor([0.1588, 2.0706, 2.0706])

    .. note::
        This implementation was adapted from the `official implementation
        <https://github.com/tum-pbs/ConFIG/tree/main/conflictfree>`_.
    """

    def __init__(self, use_least_square: bool = True):
        super().__init__()
        self.use_least_square = use_least_square

    def forward(self, matrix: Tensor) -> Tensor:
        # TODO: have a _Weighting class that does the actual computation
        weights = torch.ones(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        units = torch.nan_to_num((matrix / (matrix.norm(dim=1)).unsqueeze(1)), 0.0)
        if self.use_least_square:
            best_direction = torch.linalg.lstsq(units, weights).solution
        else:
            best_direction = torch.linalg.pinv(units) @ weights

        if best_direction.norm() == 0:
            unit_target_vector = torch.zeros_like(best_direction)
        else:
            unit_target_vector = best_direction / best_direction.norm()

        length = torch.sum(torch.stack([torch.dot(grad, unit_target_vector) for grad in matrix]))

        return length * unit_target_vector

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(use_least_square={self.use_least_square})"
