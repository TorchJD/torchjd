import pytest
import torch
from torch.nn import Linear, MSELoss, ReLU, Sequential
from torch.testing import assert_close

from torchjd import backward
from torchjd.aggregation import (
    Aggregator,
    MeanWeighting,
    MGDAWeighting,
    RandomWeighting,
    UPGradWrapper,
    WeightedAggregator,
)


@pytest.mark.parametrize(
    "A",
    [
        WeightedAggregator(MeanWeighting()),
        WeightedAggregator(UPGradWrapper(MeanWeighting())),
        WeightedAggregator(MGDAWeighting()),
        WeightedAggregator(RandomWeighting()),
    ],
)
def test_backward_various_aggregators(A: Aggregator):
    """
    Tests that backward works for various aggregators.
    """

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    backward([losses], model.parameters(), A)

    for p in model.parameters():
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_backward_valid_chunk_size(chunk_size):
    """
    Tests that backward works for various valid values of the chunk sizes parameter.
    """

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    A = WeightedAggregator(UPGradWrapper(MeanWeighting()))

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    backward([losses], model.parameters(), A, parallel_chunk_size=chunk_size)

    for p in model.parameters():
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_backward_non_positive_chunk_size(chunk_size: int):
    """
    Tests that backward raises an error when using invalid chunk sizes.
    """

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    A = WeightedAggregator(UPGradWrapper(MeanWeighting()))

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    with pytest.raises(ValueError):
        backward([losses], model.parameters(), A, parallel_chunk_size=chunk_size)


@pytest.mark.parametrize(
    "A",
    [
        WeightedAggregator(MeanWeighting()),
        WeightedAggregator(UPGradWrapper(MeanWeighting())),
        WeightedAggregator(MGDAWeighting()),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 6),
        (5, 8),
        (60, 55),
        (120, 143),
    ],
)
def test_backward_grads(A: Aggregator, shape: tuple[int]):
    """
    Tests that the .grad value filled by backward is correct in a simple example of matrix-vector
    product.
    """

    jacobian = torch.randn(shape)
    input = torch.randn([shape[1]], requires_grad=True)
    output = jacobian @ input

    backward([output], [input], A)

    assert_close(input.grad, A(jacobian))


def test_backward_empty_inputs():
    """
    Tests that backward does not fill the .grad values if no input is specified.
    """

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    A = WeightedAggregator(MeanWeighting())

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    backward([losses], [], A)

    for p in model.parameters():
        assert p.grad is None


def test_backward_partial_inputs():
    """
    Tests that backward fills the right .grad values when only a subset of the parameters are
    specified as inputs.
    """

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    A = WeightedAggregator(MeanWeighting())

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    backward([losses], model[0].parameters(), A)

    for p in model[0].parameters():
        assert (p.grad is not None) and (p.shape == p.grad.shape)

    for p in model[1:].parameters():
        assert p.grad is None


def test_backward_empty_tensors():
    """
    Tests that backward raises an error when called with an empty list of tensors.
    """

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    A = WeightedAggregator(UPGradWrapper(MeanWeighting()))

    with pytest.raises(ValueError):
        backward([], model.parameters(), A)


def test_backward_multiple_tensors():
    """
    Tests that giving multiple tensors to backward is equivalent to giving a single tensor
    containing the all the values of the original tensors.
    """

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    A = WeightedAggregator(UPGradWrapper(MeanWeighting()))

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target1 = input.sum(dim=1, keepdim=True)  # Batch of 16 targets
    target2 = torch.ones_like(target1)  # Batch of 16 other targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses1 = loss(output, target1)
    losses2 = loss(output, target2)

    backward([losses1, losses2], model.parameters(), A)

    param_to_grad = {p: p.grad for p in model.parameters()}
    for p in model.parameters():
        p.grad = None

    losses = torch.cat([losses1, losses2])
    backward(losses, model.parameters(), A)

    for p in model.parameters():
        assert (p.grad == param_to_grad[p]).all()
