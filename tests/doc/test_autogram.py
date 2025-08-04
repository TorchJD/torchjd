"""This file contains tests for the usage examples related to autogram."""


def test_augment_model_with_iwrm_backward():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd import augment_model_with_iwrm_autogram
    from torchjd.aggregation import UPGrad

    # Generate data (8 batches of 16 examples of dim 5) for the sake of the example
    inputs = torch.randn(8, 16, 5)
    targets = torch.randn(8, 16)

    model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
    optimizer = SGD(model.parameters())

    criterion = MSELoss(reduction="none")
    # TODO: improve this by making weightings public
    weighting = UPGrad().weighting.weighting
    augment_model_with_iwrm_autogram(model, weighting)

    for input, target in zip(inputs, targets):
        output = model(input)
        losses = criterion(output, target)

        optimizer.zero_grad()
        losses.backward(torch.ones_like(losses))
        optimizer.step()
