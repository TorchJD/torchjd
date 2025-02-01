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

from torchjd.aggregation._pref_vector_utils import (
    _check_pref_vector,
    _pref_vector_to_str_suffix,
    _pref_vector_to_weighting,
)
from torchjd.aggregation.bases import Aggregator
from torchjd.aggregation.sum import _SumWeighting


class ConFIG(Aggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` as defined in Equation 2 of `ConFIG: Towards
    Conflict-free Training of Physics Informed Neural Networks <https://arxiv.org/pdf/2408.11104>`_.

    :param pref_vector: The preference vector used to weight the rows. If not provided, defaults to
        equal weights of 1.

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

    def __init__(self, pref_vector: Tensor | None = None):
        super().__init__()
        _check_pref_vector(pref_vector)
        self.weighting = _pref_vector_to_weighting(pref_vector, default=_SumWeighting())
        self._pref_vector = pref_vector

    def forward(self, matrix: Tensor) -> Tensor:
        weights = self.weighting(matrix)
        units = torch.nan_to_num((matrix / (matrix.norm(dim=1)).unsqueeze(1)), 0.0)
        best_direction = torch.linalg.pinv(units) @ weights

        if best_direction.norm() == 0:
            unit_target_vector = torch.zeros_like(best_direction)
        else:
            unit_target_vector = best_direction / best_direction.norm()

        length = torch.sum(torch.stack([torch.dot(grad, unit_target_vector) for grad in matrix]))

        return length * unit_target_vector

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pref_vector={repr(self._pref_vector)})"

    def __str__(self) -> str:
        return f"ConFIG{_pref_vector_to_str_suffix(self._pref_vector)}"
