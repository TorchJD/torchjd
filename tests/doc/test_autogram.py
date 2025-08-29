"""This file contains tests for the usage examples related to autogram."""


def test_engine():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGradWeighting
    from torchjd.autogram import Engine

    # Generate data (8 batches of 16 examples of dim 5) for the sake of the example
    inputs = torch.randn(8, 16, 5)
    targets = torch.randn(8, 16, 1)

    model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
    optimizer = SGD(model.parameters())

    criterion = MSELoss(reduction="none")
    weighting = UPGradWeighting()
    engine = Engine(model.modules(), 0)

    for input, target in zip(inputs, targets):
        output = model(input)
        losses = criterion(output, target).squeeze()

        optimizer.zero_grad()
        gramian = engine.compute_gramian(losses)
        losses.backward(weighting(gramian))
        optimizer.step()
