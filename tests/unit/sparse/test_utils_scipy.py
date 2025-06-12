import importlib
import pytest
import numpy as np

scipy = pytest.importorskip("scipy")  # skip if SciPy not available
from torchjd.sparse._utils import to_coalesced_coo

def test_to_coalesced_coo_from_scipy():
    sp = importlib.import_module("scipy.sparse")
    # 2×2 off-diagonal ones
    coo = sp.coo_matrix((np.ones(2), ([0, 1], [1, 0])), shape=(2, 2))
    tsr = to_coalesced_coo(coo)  # exercises SciPy branch
    dense = tsr.to_dense()
    assert dense[0, 1] == dense[1, 0] == 1 and dense.sum() == 2
