"""Public interface for TorchJD sparse helpers.

Importing ``torchjd`` automatically activates seamless sparse support,
unless the environment variable ``TORCHJD_DISABLE_SPARSE`` is set to
``"1"`` **before** the first TorchJD import.
"""

from __future__ import annotations

import os

from ._autograd import sparse_mm  # re-export
from ._patch import enable_seamless_sparse

__all__ = ["sparse_mm"]

# feature flag
if os.getenv("TORCHJD_DISABLE_SPARSE", "0") != "1":
    enable_seamless_sparse()
