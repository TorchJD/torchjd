import warnings

from pytest import mark
from torch import Tensor, tensor
from torch.testing import assert_close

from torchjd.aggregation import (
    IMTLG,
    MGDA,
    Aggregator,
    AlignedMTL,
    CAGrad,
    ConFIG,
    Constant,
    DualProj,
    GradDrop,
    Krum,
    Mean,
    NashMTL,
    PCGrad,
    Random,
    Sum,
    TrimmedMean,
    UPGrad,
)

J_base = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
J_Krum = tensor(
    [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [75.0, -666.0, 23],  # adversarial row
        [1.0, 2.0, 3.0],
        [2.0, 0.0, 1.0],
    ]
)
J_TrimmedMean = tensor(
    [
        [1e11, 3.0],
        [1.0, -1e11],
        [-1e10, 1e10],
        [2.0, 2.0],
    ]
)

AGGREGATOR_PARAMETRIZATIONS = [
    (AlignedMTL(), J_base, tensor([0.2133, 0.9673, 0.9673])),
    (CAGrad(c=0.5), J_base, tensor([0.1835, 1.2041, 1.2041])),
    (ConFIG(), J_base, tensor([0.1588, 2.0706, 2.0706])),
    (Constant(tensor([1.0, 2.0])), J_base, tensor([8.0, 3.0, 3.0])),
    (DualProj(), J_base, tensor([0.5563, 1.1109, 1.1109])),
    (GradDrop(), J_base, tensor([6.0, 2.0, 2.0])),
    (IMTLG(), J_base, tensor([0.0767, 1.0000, 1.0000])),
    (Krum(n_byzantine=1, n_selected=4), J_Krum, tensor([1.2500, 0.7500, 1.5000])),
    (Mean(), J_base, tensor([1.0, 1.0, 1.0])),
    (MGDA(), J_base, tensor([0.0, 1.0, 1.0])),
    (NashMTL(n_tasks=2), J_base, tensor([0.0542, 0.7061, 0.7061])),
    (PCGrad(), J_base, tensor([0.5848, 3.8012, 3.8012])),
    (Random(), J_base, tensor([-2.6229, 1.0000, 1.0000])),
    (Sum(), J_base, tensor([2.0, 2.0, 2.0])),
    (TrimmedMean(trim_number=1), J_TrimmedMean, tensor([1.5000, 2.5000])),
    (UPGrad(), J_base, tensor([0.2929, 1.9004, 1.9004])),
]


@mark.parametrize(["A", "J", "expected_output"], AGGREGATOR_PARAMETRIZATIONS)
def test_aggregator_output(A: Aggregator, J: Tensor, expected_output: Tensor):
    """Test that the output values are fixed (on cpu)."""

    if isinstance(A, NashMTL):
        warnings.filterwarnings("ignore")

    assert_close(A(J), expected_output, rtol=0, atol=1e-4)
