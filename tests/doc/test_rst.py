def test_basic_usage():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    import torchjd
    from torchjd.aggregation import UPGrad

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
    optimizer = SGD(model.parameters(), lr=0.1)

    A = UPGrad()
    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target1 = torch.randn(16, 1)  # First batch of 16 targets
    target2 = torch.randn(16, 1)  # Second batch of 16 targets

    loss_fn = MSELoss()
    output = model(input)
    loss1 = loss_fn(output[:, 0], target1)
    loss2 = loss_fn(output[:, 1], target2)

    optimizer.zero_grad()
    torchjd.backward([loss1, loss2], model.parameters(), A)
    optimizer.step()


def test_iwrm():
    def test_erm_with_sgd():
        import torch
        from torch.nn import Linear, MSELoss, ReLU, Sequential
        from torch.optim import SGD

        X = torch.randn(8, 16, 10)
        Y = torch.randn(8, 16, 1)

        model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        loss_fn = MSELoss()

        params = model.parameters()
        optimizer = SGD(params, lr=0.1)

        for x, y in zip(X, Y):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_iwrm_with_ssjd():
        import torch
        from torch.nn import Linear, MSELoss, ReLU, Sequential
        from torch.optim import SGD

        from torchjd import backward
        from torchjd.aggregation import UPGrad

        X = torch.randn(8, 16, 10)
        Y = torch.randn(8, 16, 1)

        model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        loss_fn = MSELoss(reduction="none")

        params = model.parameters()
        optimizer = SGD(params, lr=0.1)
        A = UPGrad()

        for x, y in zip(X, Y):
            y_hat = model(x)
            losses = loss_fn(y_hat, y)
            optimizer.zero_grad()
            backward(losses, params, A)
            optimizer.step()

    test_erm_with_sgd()
    test_iwrm_with_ssjd()
