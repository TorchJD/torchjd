"""
This package provides implementations of several popular aggregation strategies from the literature.
The role of these aggregators is to combine a matrix (e.g. the Jacobian matrix) into a single
vector, generally to be used like a gradient.
"""

from ._aggregator_bases import Aggregator, GramianWeightedAggregator, WeightedAggregator
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
from ._weighting_bases import Matrix, PSDMatrix, Weighting

try:
    from ._cagrad import CAGrad, CAGradWeighting
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass

try:
    from ._nash_mtl import NashMTL
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass
