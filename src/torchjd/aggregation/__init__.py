from torchjd.aggregation.aligned_mtl import AlignedMTL, AlignedMTLWrapper
from torchjd.aggregation.bases import Aggregator, WeightedAggregator, Weighting
from torchjd.aggregation.cagrad import CAGrad, CAGradWeighting
from torchjd.aggregation.constant import Constant, ConstantWeighting
from torchjd.aggregation.dualproj import DualProj, DualProjWrapper
from torchjd.aggregation.graddrop import GradDrop
from torchjd.aggregation.imtl_g import IMTLG, IMTLGWeighting
from torchjd.aggregation.krum import Krum, KrumWeighting
from torchjd.aggregation.mean import Mean, MeanWeighting
from torchjd.aggregation.mgda import MGDA, MGDAWeighting
from torchjd.aggregation.nash_mtl import NashMTL, NashMTLWeighting
from torchjd.aggregation.normalizing import NormalizingWrapper
from torchjd.aggregation.pcgrad import PCGrad, PCGradWeighting
from torchjd.aggregation.random import Random, RandomWeighting
from torchjd.aggregation.sum import Sum, SumWeighting
from torchjd.aggregation.trimmed_mean import TrimmedMean
from torchjd.aggregation.upgrad import UPGrad, UPGradWrapper
