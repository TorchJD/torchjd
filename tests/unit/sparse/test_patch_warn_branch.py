"""
Covers the branch in _patch.enable_seamless_sparse() that emits a warning
when *no* ``torch_sparse`` package is available.
"""

import importlib
import sys
import types
import warnings

import torch


def test_warn_branch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_sparse", None)

    if hasattr(torch.sparse, "_orig_mm"):
        delattr(torch.sparse, "_orig_mm")  # type: ignore[attr-defined]

    import torchjd.sparse._patch as p  # noqa: E402

    p = importlib.reload(p)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        p.enable_seamless_sparse()  # <- emits RuntimeWarning branch

    assert any("SpSpMM will use slow fallback" in str(w.message) for w in rec)
