r"""
When doing Jacobian descent, the Jacobian matrix has to be aggregated into a vector to store in the
``.grad`` fields of the model parameters. The
:class:`~torchjd.aggregation._aggregator_bases.Aggregator` is responsible for these aggregations.

When using the :doc:`autogram <../autogram/index>` engine, we rather need to extract a vector
of weights from the Gramian of the Jacobian. The
:class:`~torchjd.aggregation._weighting_bases.Weighting` is responsible for this.

.. note::
    Most aggregators rely on computing the Gramian of the Jacobian, extracting a vector of weights
    from this Gramian using a :class:`~torchjd.aggregation._weighting_bases.Weighting`, and then
    combining the rows of the Jacobian using these weights. For all of them, we provide both the
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` interface (to be used in autojac) and
    the :class:`~torchjd.aggregation._weighting_bases.Weighting` interface (to be used in autogram).
    For the rest, we only provide the :class:`~torchjd.aggregation._aggregator_bases.Aggregator`
    interface -- they are not compatible with autogram.

:class:`Aggregators <torchjd.aggregation._aggregator_bases.Aggregator>` and :class:`Weightings
<torchjd.aggregation._weighting_bases.Weighting>` are callables that take a Jacobian matrix or a
Gramian matrix as inputs, respectively. The following example shows how to use UPGrad to either
aggregate a Jacobian or obtain the weights from the Gramian of the Jacobian.

>>> from torch import tensor
>>> from torchjd.aggregation import UPGrad, UPGradWeighting
>>>
>>> aggregator = UPGrad()
>>> jacobian = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
>>> aggregation = aggregator(jacobian)
>>> aggregation
tensor([0.2929, 1.9004, 1.9004])
>>> weighting = UPGradWeighting()
>>> gramian = jacobian @ jacobian.T
>>> weights = weighting(gramian)
>>> weights
tensor([1.1109, 0.7894])
"""

from ._aggregator_bases import Aggregator
from ._aligned_mtl import AlignedMTL, AlignedMTLWeighting
from ._config import ConFIG
from ._constant import Constant, ConstantWeighting
from ._dualproj import DualProj, DualProjWeighting
from ._graddrop import GradDrop
from ._imtl_g import IMTLG, IMTLGWeighting
from ._krum import Krum, KrumWeighting
from ._mean import Mean, MeanWeighting
from ._mgda import MGDA, MGDAWeighting
from ._pcgrad import PCGrad, PCGradWeighting
from ._random import Random, RandomWeighting
from ._sum import Sum, SumWeighting
from ._trimmed_mean import TrimmedMean
from ._upgrad import UPGrad, UPGradWeighting
from ._utils.check_dependencies import (
    OptionalDepsNotInstalledError as _OptionalDepsNotInstalledError,
)
from ._weighting_bases import Weighting

try:
    from ._cagrad import CAGrad, CAGradWeighting
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass

try:
    from ._nash_mtl import NashMTL
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass
