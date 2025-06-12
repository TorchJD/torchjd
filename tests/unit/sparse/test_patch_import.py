import importlib
import sys
import types
from contextlib import contextmanager


@contextmanager
def fake_torch_sparse():
    """
    Context manager that injects a *minimal* torch_sparse stub.
    The Dummy.SparseTensor *must* expose a ``matmul`` attribute because
    enable_seamless_sparse() tries to save and patch it.
    """
    mod = types.ModuleType("torch_sparse")

    class Dummy:  # noqa: D401
        # placeholder matmul so _patch can grab the attribute
        def matmul(self, dense):
            raise NotImplementedError

    mod.SparseTensor = Dummy  # type: ignore
    sys.modules["torch_sparse"] = mod
    try:
        yield
    finally:
        sys.modules.pop("torch_sparse", None)


def test_patch_without_torch_sparse(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_sparse", None)
    from importlib import reload

    import torchjd.sparse._patch as p

    reload(p)  # re-import to trigger patch
    assert p.torch_sparse is None  # slow fallback branch hit


def test_patch_with_dummy_torch_sparse(monkeypatch):
    with fake_torch_sparse():
        from importlib import reload

        import torchjd.sparse._patch as p

        reload(p)
        assert p.torch_sparse is not None  # optional branch hit
