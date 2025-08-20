"""This file contains tests for the usage examples related to autogram."""


def test_augment_model_for_iwrm():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGradWeighting
    from torchjd.autogram import GramianReverseAccumulator

    # Generate data (8 batches of 16 examples of dim 5) for the sake of the example
    inputs = torch.randn(8, 16, 5)
    targets = torch.randn(8, 16, 1)

    model = Sequential(Linear(5, 4), ReLU(), Linear(4, 1))
    optimizer = SGD(model.parameters())

    criterion = MSELoss(reduction="none")
    weighting = UPGradWeighting()
    gramian_reverse_accumulator = GramianReverseAccumulator(model.modules())

    for input, target in zip(inputs, targets):
        output = model(input)
        losses = criterion(output, target)
        # TODO: This loss is of shape [1, 16], I think it should be [16].

        optimizer.zero_grad()
        gramian = gramian_reverse_accumulator.compute_gramian(losses)
        losses.backward(weighting(gramian))
        optimizer.step()
