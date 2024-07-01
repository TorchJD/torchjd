from contextlib import nullcontext as does_not_raise

import pytest
import torch
from pytest import raises
from torch.testing import assert_close

from torchjd import multi_task_backward
from torchjd.aggregation import MGDA, Aggregator, Mean, Random, UPGrad


@pytest.mark.parametrize("A", [Mean(), UPGrad(), MGDA(), Random()])
def test_multi_task_backward_various_aggregators(A: Aggregator):
    """
    Tests that multi_task_backward works for various aggregators.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack((r1 * p1[0], r2 * p1[1]))
    y2 = r1 * p2[0] + r2 * p2[1]

    multi_task_backward(
        tasks_losses=[y1, y2],
        shared_parameters=[p0],
        shared_representations=[r1, r2],
        tasks_parameters=[[p1], [p2]],
        A=A,
    )

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@pytest.mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_multi_task_backward_valid_chunk_size(chunk_size):
    """
    Tests that multi_task_backward works for various valid values of the chunk sizes parameter.
    """

    A = UPGrad()

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack((r1 * p1[0], r2 * p1[1]))
    y2 = r1 * p2[0] + r2 * p2[1]

    multi_task_backward(
        tasks_losses=[y1, y2],
        shared_parameters=[p0],
        shared_representations=[r1, r2],
        tasks_parameters=[[p1], [p2]],
        A=A,
        parallel_chunk_size=chunk_size,
        retain_graph=True,
    )

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_multi_task_backward_non_positive_chunk_size(chunk_size: int):
    """
    Tests that multi_task_backward raises an error when using invalid chunk sizes.
    """

    A = UPGrad()

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack((r1 * p1[0], r2 * p1[1]))
    y2 = r1 * p2[0] + r2 * p2[1]

    with pytest.raises(ValueError):
        multi_task_backward(
            tasks_losses=[y1, y2],
            shared_parameters=[p0],
            shared_representations=[r1, r2],
            tasks_parameters=[[p1], [p2]],
            A=A,
            parallel_chunk_size=chunk_size,
        )


@pytest.mark.parametrize(
    ["chunk_size", "expectation"],
    [(1, raises(ValueError)), (2, does_not_raise()), (None, does_not_raise())],
)
def test_multi_task_backward_no_retain_graph_small_chunk_size(chunk_size: int, expectation):
    """
    Tests that multi_task_backward raises an error when using retain_graph=False and a chunk size
    that is not large enough to allow differentiation of all tensors are once.
    """

    A = UPGrad()

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack((r1 * p1[0], r2 * p1[1]))
    y2 = r1 * p2[0] + r2 * p2[1]

    with expectation:
        multi_task_backward(
            tasks_losses=[y1, y2],
            shared_parameters=[p0],
            shared_representations=[r1, r2],
            tasks_parameters=[[p1], [p2]],
            A=A,
            parallel_chunk_size=chunk_size,
        )


@pytest.mark.parametrize("A", [Mean(), UPGrad(), MGDA()])
@pytest.mark.parametrize("shape", [(2, 3), (2, 6), (5, 8), (60, 55), (120, 143)])
def test_multi_task_backward_value_is_correct(A: Aggregator, shape: tuple[int]):
    """
    Tests that the .grad value filled by multi_task_backward is correct in a simple example of
    matrix-vector product for shared representation and three tasks whose loss are given by a simple
    inner product of the shared representation with the task parameter.
    """

    J = torch.randn(shape)
    input = torch.randn([shape[1]], requires_grad=True)

    r = J @ input

    p1 = torch.randn(shape[0], requires_grad=True)
    p2 = torch.randn(shape[0], requires_grad=True)
    p3 = torch.randn(shape[0], requires_grad=True)

    y1 = p1 @ r
    y2 = p2 @ r
    y3 = p3 @ r

    multi_task_backward(
        tasks_losses=[y1, y2, y3],
        shared_parameters=[input],
        shared_representations=r,
        tasks_parameters=[[p1], [p2], [p3]],
        A=A,
    )

    assert_close(p1.grad, r)
    assert_close(p2.grad, r)
    assert_close(p3.grad, r)

    resulting_J = torch.stack((p1, p2, p3)) @ J
    assert_close(input.grad, A(resulting_J))


def test_multi_task_backward_empty_parameters():
    """
    Tests that multi_task_backward does not fill the .grad values if no input is specified.
    """

    A = UPGrad()

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack((r1 * p1[0], r2 * p1[1]))
    y2 = r1 * p2[0] + r2 * p2[1]

    multi_task_backward(
        tasks_losses=[y1, y2],
        shared_parameters=[],
        shared_representations=[r1, r2],
        tasks_parameters=[[], []],
        A=A,
    )

    for p in [p0, p1, p2]:
        assert p.grad is None


def test_multi_task_backward_partial_parameters():
    """
    Tests that multi_task_backward fills the right .grad values when only a subset of the parameters
    are specified as inputs.
    """

    A = Mean()

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack((r1 * p1[0], r2 * p1[1]))
    y2 = r1 * p2[0] + r2 * p2[1]

    multi_task_backward(
        tasks_losses=[y1, y2],
        shared_parameters=[p0],
        shared_representations=[r1, r2],
        tasks_parameters=[[p1], []],
        A=A,
    )

    assert (p0.grad is not None) and (p0.shape == p0.grad.shape)
    assert (p1.grad is not None) and (p1.shape == p1.grad.shape)
    assert p2.grad is None


def test_multi_task_backward_empty_tasks():
    """
    Tests that multi_task_backward raises an error when called with an empty list of tasks.
    """

    A = UPGrad()

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)

    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()

    with pytest.raises(ValueError):
        multi_task_backward(
            tasks_losses=[],
            shared_parameters=[p0],
            shared_representations=[r1, r2],
            tasks_parameters=[],
            A=A,
        )


def test_multi_task_backward_incoherent_task_number():
    """
    Tests that multi_task_backward raises an error when called with the number of tasks losses
    different from the number of tasks parameters.
    """

    A = UPGrad()

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    r1 = torch.tensor([-1.0, 1.0]) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack((r1 * p1[0], r2 * p1[1]))
    y2 = r1 * p2[0] + r2 * p2[1]

    with pytest.raises(ValueError):
        multi_task_backward(
            tasks_losses=[y1, y2],
            shared_parameters=[p0],
            shared_representations=[r1, r2],
            tasks_parameters=[[p1]],  # Wrong
            A=A,
            retain_graph=True,
        )
    with pytest.raises(ValueError):
        multi_task_backward(
            tasks_losses=[y1],  # Wrong
            shared_parameters=[p0],
            shared_representations=[r1, r2],
            tasks_parameters=[[p1], [p2]],
            A=A,
            retain_graph=True,
        )
