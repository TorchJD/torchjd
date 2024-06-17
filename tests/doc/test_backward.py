def test_backward():
    import torch

    from torchjd import backward
    from torchjd.aggregation import UPGrad

    param = torch.tensor([1.0, 2.0], requires_grad=True)
    # Compute arbitrary quantities that are function of param
    y1 = torch.tensor([-1.0, 1.0]) @ param
    y2 = (param**2).sum()

    backward([y1, y2], [param], A=UPGrad())
