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
    target1 = torch.randn(16)  # First batch of 16 targets
    target2 = torch.randn(16)  # Second batch of 16 targets

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


def test_mtl():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd import mtl_backward
    from torchjd.aggregation import UPGrad

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]

    loss_fn = MSELoss()
    optimizer = SGD(params, lr=0.1)
    A = UPGrad()

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 input random vectors of length 10
    task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        features = shared_module(input)
        output1 = task1_module(features)
        output2 = task2_module(features)
        loss1 = loss_fn(output1, target1)
        loss2 = loss_fn(output2, target2)

        optimizer.zero_grad()
        mtl_backward(
            features=features,
            losses=[loss1, loss2],
            shared_params=shared_module.parameters(),
            tasks_params=[task1_module.parameters(), task2_module.parameters()],
            A=A,
        )
        optimizer.step()
