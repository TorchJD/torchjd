from contextlib import nullcontext as does_not_raise

import pytest
import torch
from pytest import raises
from torch.testing import assert_close
from unit._utils import ExceptionContext
from unit.conftest import DEVICE

from torchjd import backward
from torchjd.aggregation import MGDA, Aggregator, Mean, Random, UPGrad


@pytest.mark.parametrize("A", [Mean(), UPGrad(), MGDA(), Random()])
def test_backward_various_aggregators(A: Aggregator):
    """Tests that backward works for various aggregators."""

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    params = [p1, p2]

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    backward([y1, y2], params, A)

    for p in params:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@pytest.mark.parametrize("A", [Mean(), UPGrad(), MGDA()])
@pytest.mark.parametrize("shape", [(2, 3), (2, 6), (5, 8), (60, 55), (120, 143)])
def test_backward_value_is_correct_1(A: Aggregator, shape: tuple[int, int]):
    """
    Tests that the .grad value filled by backward is correct in a simple example of matrix-vector
    product.
    """

    J = torch.randn(shape, device=DEVICE)
    input = torch.randn([shape[1]], requires_grad=True, device=DEVICE)
    output = J @ input  # Note that the Jacobian of output w.r.t. input is J.

    backward([output], [input], A)

    assert_close(input.grad, A(J))


@pytest.mark.parametrize("A", [Mean(), UPGrad(), MGDA()])
@pytest.mark.parametrize("shape", [(2, 3), (2, 6), (5, 8), (60, 55), (120, 143)])
def test_backward_value_is_correct_2(A: Aggregator, shape: tuple[int, int]):
    """
    Tests that the .grad value filled by backward is correct in a simple example of matrix-vector
    product.
    """

    p1 = torch.randn([shape[1]], requires_grad=True, device=DEVICE)
    p2 = torch.randn([shape[1]], requires_grad=True, device=DEVICE)

    J1 = torch.randn(shape, device=DEVICE)
    J2 = torch.randn(shape, device=DEVICE)
    J3 = torch.randn(shape, device=DEVICE)
    y1 = J1 @ p1 + J2 @ p2
    y2 = J3 @ p2

    backward(tensors=[y1, y2], A=A, inputs=[p1, p2])

    zeros = torch.zeros_like(J1)

    expected_jacobian = torch.vstack([torch.hstack([J1, J2]), torch.hstack([zeros, J3])])
    # J1 J2
    # 0  J3

    expected_aggregation = A(expected_jacobian)

    assert_close(p1.grad, expected_aggregation[: shape[0]])
    assert_close(p2.grad, expected_aggregation[shape[0] :])


def test_backward_empty_inputs():
    """Tests that backward does not fill the .grad values if no input is specified."""

    A = Mean()

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    params = [p1, p2]

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    backward([y1, y2], [], A)

    for p in params:
        assert p.grad is None


def test_backward_partial_inputs():
    """
    Tests that backward fills the right .grad values when only a subset of the parameters are
    specified as inputs.
    """

    A = Mean()

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    backward([y1, y2], [p1], A)

    assert (p1.grad is not None) and (p1.shape == p1.grad.shape)
    assert p2.grad is None


def test_backward_empty_tensors():
    """Tests that backward raises an error when called with an empty list of tensors."""

    A = UPGrad()

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    with pytest.raises(ValueError):
        backward([], [p1, p2], A)


def test_backward_multiple_tensors():
    """
    Tests that giving multiple tensors to backward is equivalent to giving a single tensor
    containing the all the values of the original tensors.
    """

    A = UPGrad()

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    params = [p1, p2]

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    backward([y1, y2], params, A, retain_graph=True)

    param_to_grad = {p: p.grad for p in params}
    for p in params:
        p.grad = None

    backward(torch.cat([y1.reshape(-1), y2.reshape(-1)]), params, A)

    for p in params:
        assert (p.grad == param_to_grad[p]).all()


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_backward_valid_chunk_size(chunk_size):
    """Tests that backward works for various valid values of parallel_chunk_size."""

    A = UPGrad()

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    params = [p1, p2]

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    backward([y1, y2], params, A, parallel_chunk_size=chunk_size, retain_graph=True)

    for p in params:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_backward_non_positive_chunk_size(chunk_size: int):
    """Tests that backward raises an error when using invalid chunk sizes."""

    A = UPGrad()

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    params = [p1, p2]

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    with pytest.raises(ValueError):
        backward([y1, y2], params, A, parallel_chunk_size=chunk_size)


@pytest.mark.parametrize(
    ["chunk_size", "expectation"],
    [(1, raises(ValueError)), (2, does_not_raise()), (None, does_not_raise())],
)
def test_backward_no_retain_graph_small_chunk_size(chunk_size: int, expectation: ExceptionContext):
    """
    Tests that backward raises an error when using retain_graph=False and a chunk size that is not
    large enough to allow differentiation of all tensors at once.
    """

    A = UPGrad()

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    params = [p1, p2]

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    with expectation:
        backward([y1, y2], params, A, retain_graph=False, parallel_chunk_size=chunk_size)
