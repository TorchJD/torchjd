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
    "aggregator",
    [
        WeightedAggregator(MeanWeighting()),
        WeightedAggregator(UPGradWrapper(MeanWeighting())),
        WeightedAggregator(MGDAWeighting()),
        WeightedAggregator(RandomWeighting()),
    ],
)
def test_backward_various_aggregators(aggregator: Aggregator):
    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    backward([losses], model.parameters(), aggregator)

    for p in model.parameters():
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_backward_valid_chunk_size(chunk_size):
    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))

    W = UPGradWrapper(MeanWeighting())
    A = WeightedAggregator(W)

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
    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))

    W = UPGradWrapper(MeanWeighting())
    A = WeightedAggregator(W)

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    with pytest.raises(ValueError):
        backward([losses], model.parameters(), A, parallel_chunk_size=chunk_size)


@pytest.mark.parametrize(
    "aggregator",
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
def test_backward_grads(aggregator: Aggregator, shape: tuple[int]):
    jacobian = torch.randn(shape)
    input = torch.randn([shape[1]], requires_grad=True)
    output = jacobian @ input

    backward([output], [input], aggregator)

    assert_close(input.grad, aggregator(jacobian))


def test_backward_empty_inputs():
    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    aggregator = WeightedAggregator(MeanWeighting())

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    backward([losses], [], aggregator)

    for p in model.parameters():
        assert p.grad is None


def test_backward_partial_inputs():
    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    aggregator = WeightedAggregator(MeanWeighting())

    input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
    target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

    loss = MSELoss(reduction="none")

    output = model(input)
    losses = loss(output, target)

    backward([losses], model[0].parameters(), aggregator)

    for p in model[0].parameters():
        assert (p.grad is not None) and (p.shape == p.grad.shape)

    for p in model[1:].parameters():
        assert p.grad is None


def test_backward_empty_tensors():
    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))

    W = UPGradWrapper(MeanWeighting())
    A = WeightedAggregator(W)

    with pytest.raises(ValueError):
        backward([], model.parameters(), A)
