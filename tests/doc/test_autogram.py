"""This file contains tests for the usage examples related to autogram."""

from torchjd.aggregation import UPGradWeighting


def test_augment_model_for_gramian_based_iwrm():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd import augment_model_for_gramian_based_iwrm

    # Generate data (8 batches of 16 examples of dim 5) for the sake of the example
    inputs = torch.randn(8, 16, 5)
    targets = torch.randn(8, 16)

    model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
    optimizer = SGD(model.parameters())

    criterion = MSELoss(reduction="none")
    weighting = UPGradWeighting()
    augment_model_for_gramian_based_iwrm(model, weighting)

    for input, target in zip(inputs, targets):
        output = model(input)
        losses = criterion(output, target)

        optimizer.zero_grad()
        losses.backward(torch.ones_like(losses))
        optimizer.step()
