"""
This package provides implementations of several popular aggregation strategies from the literature.
The role of these aggregators is to combine a matrix (e.g. the Jacobian matrix) into a single
vector, generally to be used like a gradient.
"""

from ._aggregator_bases import Aggregator
from ._aligned_mtl import AlignedMTL
from ._config import ConFIG
from ._constant import Constant
from ._dualproj import DualProj
from ._graddrop import GradDrop
from ._imtl_g import IMTLG
from ._krum import Krum
from ._mean import Mean
from ._mgda import MGDA
from ._pcgrad import PCGrad
from ._random import Random
from ._sum import Sum
from ._trimmed_mean import TrimmedMean
from ._upgrad import UPGrad
from ._utils.check_dependencies import (
    OptionalDepsNotInstalledError as _OptionalDepsNotInstalledError,
)

try:
    from ._cagrad import CAGrad
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass

try:
    from ._nash_mtl import NashMTL
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass
