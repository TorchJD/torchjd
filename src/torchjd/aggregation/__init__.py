from torchjd.aggregation._utils.check_dependencies import (
    OptionalDepsNotInstalledError as _OptionalDepsNotInstalledError,
)

from .aligned_mtl import AlignedMTL
from .bases import Aggregator
from .config import ConFIG
from .constant import Constant
from .dualproj import DualProj
from .graddrop import GradDrop
from .imtl_g import IMTLG
from .krum import Krum
from .mean import Mean
from .mgda import MGDA
from .pcgrad import PCGrad
from .random import Random
from .sum import Sum
from .trimmed_mean import TrimmedMean
from .upgrad import UPGrad

try:
    from .cagrad import CAGrad
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass

try:
    from .nash_mtl import NashMTL
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass
