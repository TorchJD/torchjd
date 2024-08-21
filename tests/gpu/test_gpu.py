import torch.cuda


def test_gpu():
    assert torch.cuda.is_available()
    t = torch.tensor(10.0, device="cuda")
    assert t is not None
