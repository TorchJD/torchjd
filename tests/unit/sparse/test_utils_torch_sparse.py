import importlib, sys, types, torch

def test_to_coalesced_coo_torch_sparse(monkeypatch):
    dummy = types.ModuleType("torch_sparse")

    class DummyTensor:  # noqa: D401
        def __init__(self):
            self.row = torch.tensor([0])
            self.col = torch.tensor([0])
            self.value = torch.tensor([1.0])

        def to_torch_sparse_coo_tensor(self):
            return torch.sparse_coo_tensor(
                torch.stack([self.row, self.col]), self.value, (1, 1)
            )

        def matmul(self, other):
            raise NotImplementedError

    dummy.SparseTensor = DummyTensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_sparse", dummy)

    utils = importlib.reload(importlib.import_module("torchjd.sparse._utils"))
    to_coalesced_coo = utils.to_coalesced_coo

    tsr = to_coalesced_coo(DummyTensor())
    assert tsr.is_sparse and tsr._nnz() == 1
