def test_basic_usage():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    import torchjd
    from torchjd.aggregation import UPGrad

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    optimizer = SGD(model.parameters(), lr=0.1)

    A = UPGrad()

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss_fn = MSELoss(reduction="none")
    output = model(input)
    losses = loss_fn(output, target)

    optimizer.zero_grad()
    torchjd.backward(losses, model.parameters(), A)
    optimizer.step()


def test_ssjd():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    import torchjd
    from torchjd.aggregation import UPGrad

    A = UPGrad()

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    loss_fn = MSELoss(reduction="none")
    params = model.parameters()
    optimizer = SGD(params, lr=0.1)

    x = torch.randn(16, 10)
    y = torch.randn(16, 1)
    y_hat = model(x)
    losses = loss_fn(y_hat, y)

    optimizer.zero_grad()
    torchjd.backward(losses, params, A)
    optimizer.step()
