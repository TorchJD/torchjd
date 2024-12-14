from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch.testing import assert_close
from unit._utils import ExceptionContext
from unit.conftest import DEVICE

from torchjd import backward
from torchjd.aggregation import MGDA, Aggregator, Mean, Random, UPGrad


@mark.parametrize("aggregator", [Mean(), UPGrad(), MGDA(), Random()])
def test_various_aggregators(aggregator: Aggregator):
    """Tests that backward works for various aggregators."""

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], aggregator)

    for a in [a1, a2]:
        assert (a.grad is not None) and (a.shape == a.grad.shape)


@mark.parametrize("aggregator", [Mean(), UPGrad(), MGDA()])
@mark.parametrize("shape", [(2, 3), (2, 6), (5, 8), (60, 55), (120, 143)])
@mark.parametrize("manually_specify_inputs", [True, False])
def test_value_is_correct(
    aggregator: Aggregator, shape: tuple[int, int], manually_specify_inputs: bool
):
    """
    Tests that the .grad value filled by backward is correct in a simple example of matrix-vector
    product.
    """

    J = torch.randn(shape, device=DEVICE)
    input = torch.randn([shape[1]], requires_grad=True, device=DEVICE)
    output = J @ input  # Note that the Jacobian of output w.r.t. input is J.

    if manually_specify_inputs:
        inputs = [input]
    else:
        inputs = None

    backward([output], aggregator, inputs=inputs)

    assert_close(input.grad, aggregator(J))


def test_empty_inputs():
    """Tests that backward does not fill the .grad values if no input is specified."""

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], Mean(), inputs=[])

    for a in [a1, a2]:
        assert a.grad is None


def test_partial_inputs():
    """
    Tests that backward fills the right .grad values when only a subset of the actual inputs are
    specified as inputs.
    """

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], Mean(), inputs=[a1])

    assert (a1.grad is not None) and (a1.shape == a1.grad.shape)
    assert a2.grad is None


def test_empty_tensors_fails():
    """Tests that backward raises an error when called with an empty list of tensors."""

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    with raises(ValueError):
        backward([], UPGrad(), inputs=[a1, a2])


def test_multiple_tensors():
    """
    Tests that giving multiple tensors to backward is equivalent to giving a single tensor
    containing the all the values of the original tensors.
    """

    aggregator = UPGrad()

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    inputs = [a1, a2]

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], aggregator, retain_graph=True)

    input_to_grad = {a: a.grad for a in inputs}
    for a in inputs:
        a.grad = None

    backward(torch.cat([y1.reshape(-1), y2.reshape(-1)]), aggregator)

    for a in inputs:
        assert (a.grad == input_to_grad[a]).all()


@mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_various_valid_chunk_sizes(chunk_size):
    """Tests that backward works for various valid values of parallel_chunk_size."""

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    backward([y1, y2], UPGrad(), parallel_chunk_size=chunk_size, retain_graph=True)

    for a in [a1, a2]:
        assert (a.grad is not None) and (a.shape == a.grad.shape)


@mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_fails(chunk_size: int):
    """Tests that backward raises an error when using invalid chunk sizes."""

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    with raises(ValueError):
        backward([y1, y2], UPGrad(), parallel_chunk_size=chunk_size)


@mark.parametrize(
    ["chunk_size", "expectation"],
    [(1, raises(ValueError)), (2, does_not_raise()), (None, does_not_raise())],
)
def test_no_retain_graph_various_chunk_sizes(chunk_size: int, expectation: ExceptionContext):
    """
    Tests that when using retain_graph=False, backward only works if the chunk size is large enough
    to allow differentiation of all tensors at once.
    """

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    with expectation:
        backward([y1, y2], UPGrad(), retain_graph=False, parallel_chunk_size=chunk_size)


def test_input_retaining_grad_fails():
    """
    Tests that backward raises an error when some input in the computation graph of the ``tensors``
    parameter retains grad.
    """

    a = torch.tensor(1.0, requires_grad=True, device=DEVICE)
    b = 2 * a
    b.retain_grad()
    y = 3 * b

    with raises(RuntimeError):
        backward(tensors=y, aggregator=UPGrad(), inputs=[b])


def test_non_input_retaining_grad_fails():
    """
    Tests that backward fails to fill a valid `.grad` when some tensor in the computation graph of
    the ``tensors`` parameter retains grad.
    """

    a = torch.tensor(1.0, requires_grad=True, device=DEVICE)
    b = 2 * a
    b.retain_grad()
    y = 3 * b

    # backward itself doesn't raise the error, but it fills b.grad with a BatchedTensor
    backward(tensors=y, aggregator=UPGrad(), inputs=[a])

    with raises(RuntimeError):
        # Using such a BatchedTensor should result in an error
        _ = -b.grad
