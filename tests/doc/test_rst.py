def test_dummy():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    import torchjd
    from torchjd.aggregation import MeanWeighting, UPGradWrapper, WeightedAggregator

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    optimizer = SGD(model.parameters(), lr=0.1)

    W = UPGradWrapper(MeanWeighting())
    A = WeightedAggregator(W)

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(axis=1, keepdim=True)  # Batch of 16 targets
    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    optimizer.zero_grad()

    torchjd.backward(losses, model.parameters(), A)
    optimizer.step()
