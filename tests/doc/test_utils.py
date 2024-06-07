def test_backward():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential

    from torchjd import backward
    from torchjd.aggregation import MeanWeighting, UPGradWrapper, WeightedAggregator

    _ = torch.manual_seed(0)  # Set the seed to make this example deterministic

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    loss = MSELoss(reduction="none")

    W = UPGradWrapper(MeanWeighting())
    A = WeightedAggregator(W)

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    output = model(input)
    losses = loss(output, target)

    backward(losses, model.parameters(), A)
