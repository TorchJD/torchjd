import importlib
import sys
import types
from importlib import reload


def _make_dummy_torch_sparse():
    """
    Return a minimal torch_sparse stub:

    * SparseTensor.matmul  – so _patch can save & wrap it.
    * SparseTensor.to_torch_sparse_coo_tensor – so _utils branch works.
    """
    dummy_mod = types.ModuleType("torch_sparse")

    class DummyTensor:  # noqa: D401
        def matmul(self, dense):
            raise NotImplementedError

        def to_torch_sparse_coo_tensor(self):
            import torch

            return torch.sparse_coo_tensor([[0], [0]], [1.0], (1, 1))

    dummy_mod.SparseTensor = DummyTensor  # type: ignore[attr-defined]
    return dummy_mod


def test_full_torch_sparse_branch(monkeypatch):
    # Inject fresh stub
    monkeypatch.setitem(sys.modules, "torch_sparse", _make_dummy_torch_sparse())

    # Force the patch module to re-evaluate from scratch
    # Remove earlier sentinel attributes so enable_seamless_sparse() re-patches
    import torch

    import torchjd.sparse._patch as p  # noqa: E402

    for attr in ("_orig_mm",):
        if hasattr(torch.sparse, attr):
            delattr(torch.sparse, attr)  # type: ignore[attr-defined]

    # Run patch
    reload(p)
    p.enable_seamless_sparse()

    # Optional branch should have set _orig_matmul
    assert hasattr(p.torch_sparse.SparseTensor, "_orig_matmul")
