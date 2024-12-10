from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch.testing import assert_close
from unit._utils import ExceptionContext
from unit.conftest import DEVICE

from torchjd import mtl_backward
from torchjd.aggregation import MGDA, Aggregator, Mean, Random, UPGrad


@mark.parametrize("aggregator", [Mean(), UPGrad(), MGDA(), Random()])
def test_mtl_backward_various_aggregators(aggregator: Aggregator):
    """Tests that mtl_backward works for various aggregators."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]
    y2 = r1 * p2[0] + r2 * p2[1]

    mtl_backward(losses=[y1, y2], features=[r1, r2], aggregator=aggregator)

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@mark.parametrize("aggregator", [Mean(), UPGrad(), MGDA()])
@mark.parametrize("shape", [(2, 3), (2, 6), (5, 8), (60, 55), (120, 143)])
@mark.parametrize("manually_specify_shared_params", [True, False])
@mark.parametrize("manually_specify_tasks_params", [True, False])
def test_mtl_backward_value_is_correct(
    aggregator: Aggregator,
    shape: tuple[int, int],
    manually_specify_shared_params: bool,
    manually_specify_tasks_params: bool,
):
    """
    Tests that the .grad value filled by mtl_backward is correct in a simple example of
    matrix-vector product for shared representation and three tasks whose loss are given by a simple
    inner product of the shared representation with the task parameter.

    This test should work with or without manually specifying the parameters.
    """

    p0 = torch.randn([shape[1]], requires_grad=True, device=DEVICE)
    p1 = torch.randn(shape[0], requires_grad=True, device=DEVICE)
    p2 = torch.randn(shape[0], requires_grad=True, device=DEVICE)
    p3 = torch.randn(shape[0], requires_grad=True, device=DEVICE)

    J = torch.randn(shape, device=DEVICE)
    r = J @ p0
    y1 = p1 @ r
    y2 = p2 @ r
    y3 = p3 @ r

    if manually_specify_shared_params:
        shared_params = [p0]
    else:
        shared_params = None

    if manually_specify_tasks_params:
        tasks_params = [[p1], [p2], [p3]]
    else:
        tasks_params = None

    mtl_backward(
        losses=[y1, y2, y3],
        features=r,
        aggregator=aggregator,
        tasks_params=tasks_params,
        shared_params=shared_params,
    )

    assert_close(p1.grad, r)
    assert_close(p2.grad, r)
    assert_close(p3.grad, r)

    expected_jacobian = torch.stack((p1, p2, p3)) @ J
    expected_aggregation = aggregator(expected_jacobian)

    assert_close(p0.grad, expected_aggregation)


def test_mtl_backward_empty_tasks():
    """Tests that mtl_backward raises an error when called with an empty list of tasks."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()

    with raises(ValueError):
        mtl_backward(losses=[], features=[r1, r2], aggregator=UPGrad())


def test_mtl_backward_single_task():
    """Tests that mtl_backward works correctly with a single task."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]

    mtl_backward(losses=[y1], features=[r1, r2], aggregator=UPGrad())

    for p in [p0, p1]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


def test_mtl_backward_incoherent_task_number():
    """
    Tests that mtl_backward raises an error when called with the number of tasks losses different
    from the number of tasks parameters.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]
    y2 = r1 * p2[0] + r2 * p2[1]

    with raises(ValueError):
        mtl_backward(
            losses=[y1, y2],
            features=[r1, r2],
            aggregator=UPGrad(),
            tasks_params=[[p1]],  # Wrong
            shared_params=[p0],
        )
    with raises(ValueError):
        mtl_backward(
            losses=[y1],  # Wrong
            features=[r1, r2],
            aggregator=UPGrad(),
            tasks_params=[[p1], [p2]],
            shared_params=[p0],
        )


def test_mtl_backward_empty_params():
    """Tests that mtl_backward does not fill the .grad values if no parameter is specified."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]
    y2 = r1 * p2[0] + r2 * p2[1]

    mtl_backward(
        losses=[y1, y2],
        features=[r1, r2],
        aggregator=UPGrad(),
        tasks_params=[[], []],
        shared_params=[],
    )

    for p in [p0, p1, p2]:
        assert p.grad is None


def test_mtl_backward_multiple_params_per_task():
    """Tests that mtl_backward works correctly when the tasks each have several parameters."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1_a = torch.tensor(1.0, requires_grad=True, device=DEVICE)
    p1_b = torch.tensor([2.0, 3.0], requires_grad=True, device=DEVICE)
    p1_c = torch.tensor([[4.0, 5.0], [6.0, 7.0]], requires_grad=True, device=DEVICE)
    p2_a = torch.tensor(8.0, requires_grad=True, device=DEVICE)
    p2_b = torch.tensor([9.0, 10.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1_a + (r2 * p1_b).sum() + (r1 * p1_c).sum()
    y2 = r1 * p2_a * (r2 * p2_b).sum()

    mtl_backward(losses=[y1, y2], features=[r1, r2], aggregator=UPGrad())

    for p in [p0, p1_a, p1_b, p1_c, p2_a, p2_b]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@mark.parametrize(
    "shared_params_shapes",
    [
        [tuple()],
        [(2,)],
        [(3, 2)],
        [(4, 3, 2)],
        [tuple(), (2,)],
        [(3, 2), (2,)],
        [(4, 3, 2), (3, 2), tuple()],
        [(5, 4, 3, 2), (5, 4, 3, 2)],
    ],
)
def test_mtl_backward_various_shared_params(shared_params_shapes: list[tuple[int]]):
    """Tests that mtl_backward works correctly with various kinds of shared_params."""

    shared_params = [
        torch.rand(shape, requires_grad=True, device=DEVICE) for shape in shared_params_shapes
    ]
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    representations = [shared_param.sum(dim=-1) for shared_param in shared_params]
    y1 = torch.stack([r.sum() for r in representations]).sum()
    y2 = torch.stack([r.sum() ** 2 for r in representations]).sum()

    mtl_backward(
        losses=[y1, y2],
        features=representations,
        aggregator=UPGrad(),
        tasks_params=[[p1], [p2]],  # Enforce differentiation w.r.t. params that haven't been used
        shared_params=shared_params,
    )

    for p in [*shared_params, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


def test_mtl_backward_partial_params():
    """
    Tests that mtl_backward fills the right .grad values when only a subset of the parameters are
    specified as inputs.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]
    y2 = r1 * p2[0] + r2 * p2[1]

    mtl_backward(
        losses=[y1, y2],
        features=[r1, r2],
        aggregator=Mean(),
        tasks_params=[[p1], []],
        shared_params=[p0],
    )

    assert (p0.grad is not None) and (p0.shape == p0.grad.shape)
    assert (p1.grad is not None) and (p1.shape == p1.grad.shape)
    assert p2.grad is None


def test_mtl_backward_empty_features():
    """Tests that mtl_backward expectedly raises an error when no there is no feature."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]
    y2 = r1 * p2[0] + r2 * p2[1]

    with raises(ValueError):
        mtl_backward(losses=[y1, y2], features=[], aggregator=UPGrad())


@mark.parametrize(
    "shape",
    [
        (2,),
        (3, 2),
        (4, 3, 2),
        (5, 4, 3, 2),
    ],
)
def test_mtl_backward_various_single_features(shape: tuple[int, ...]):
    """Tests that mtl_backward works correctly with various kinds of feature tensors."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([5.0, 6.0], requires_grad=True, device=DEVICE)

    r = torch.rand(shape, device=DEVICE) @ p0

    y1 = (r * p1[0]).sum() + (r * p1[1]).sum()
    y2 = (r * p2[0]).sum() * (r * p2[1]).sum()

    mtl_backward(losses=[y1, y2], features=r, aggregator=UPGrad())

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@mark.parametrize(
    "shapes",
    [
        [(2,)],
        [(3, 2)],
        [(4, 3, 2)],
        [(5, 4, 3, 2)],
        [(2,), (2,)],
        [(3, 2), (2,)],
        [(4, 3, 2), (3, 2), (2,)],
        [(5, 4, 3, 2), (5, 4, 3, 2)],
    ],
)
def test_mtl_backward_various_feature_lists(shapes: list[tuple[int]]):
    """Tests that mtl_backward works correctly with various kinds of feature lists."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.arange(len(shapes), dtype=torch.float32, requires_grad=True, device=DEVICE)
    p2 = torch.tensor(5.0, requires_grad=True, device=DEVICE)

    representations = [torch.rand(shape, device=DEVICE) @ p0 for shape in shapes]

    y1 = sum([(r * p).sum() for r, p in zip(representations, p1)])
    y2 = (representations[0] * p2).sum()

    mtl_backward(losses=[y1, y2], features=representations, aggregator=UPGrad())

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


def test_mtl_backward_non_scalar_loss():
    """Tests that mtl_backward raises an error when used with a non-scalar loss."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack([r1 * p1[0], r2 * p1[1]])  # Non-scalar
    y2 = r1 * p2[0] + r2 * p2[1]

    with raises(ValueError):
        mtl_backward(losses=[y1, y2], features=[r1, r2], aggregator=UPGrad())


@mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_mtl_backward_valid_chunk_size(chunk_size):
    """Tests that mtl_backward works for various valid values of parallel_chunk_size."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]
    y2 = r1 * p2[0] + r2 * p2[1]

    mtl_backward(
        losses=[y1, y2],
        features=[r1, r2],
        aggregator=UPGrad(),
        retain_graph=True,
        parallel_chunk_size=chunk_size,
    )

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@mark.parametrize("chunk_size", [0, -1])
def test_mtl_backward_non_positive_chunk_size(chunk_size: int):
    """Tests that mtl_backward raises an error when using invalid chunk sizes."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]
    y2 = r1 * p2[0] + r2 * p2[1]

    with raises(ValueError):
        mtl_backward(
            losses=[y1, y2],
            features=[r1, r2],
            aggregator=UPGrad(),
            parallel_chunk_size=chunk_size,
        )


@mark.parametrize(
    ["chunk_size", "expectation"],
    [(1, raises(ValueError)), (2, does_not_raise()), (None, does_not_raise())],
)
def test_mtl_backward_no_retain_graph_small_chunk_size(
    chunk_size: int, expectation: ExceptionContext
):
    """
    Tests that mtl_backward raises an error when using retain_graph=False and a chunk size that is
    not large enough to allow differentiation of all tensors at once.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    r1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    r2 = (p0**2).sum() + p0.norm()
    y1 = r1 * p1[0] + r2 * p1[1]
    y2 = r1 * p2[0] + r2 * p2[1]

    with expectation:
        mtl_backward(
            losses=[y1, y2],
            features=[r1, r2],
            aggregator=UPGrad(),
            retain_graph=False,
            parallel_chunk_size=chunk_size,
        )


def test_mtl_backward_fails_with_shared_param_retaining_grad():
    """
    Tests that mtl_backward raises an error when some shared param in the computation graph of the
    ``features`` parameter retains grad.
    """

    p0 = torch.tensor(1.0, requires_grad=True, device=DEVICE)
    p1 = torch.tensor(2.0, requires_grad=True, device=DEVICE)
    p2 = torch.tensor(3.0, requires_grad=True, device=DEVICE)

    a = 2 * p0
    a.retain_grad()
    features = 3 * a
    y1 = p1 * features
    y2 = p2 * features

    with raises(RuntimeError):
        mtl_backward(
            losses=[y1, y2],
            features=[features],
            aggregator=UPGrad(),
            tasks_params=[[p1], [p2]],
            shared_params=[a, p0],
        )


def test_mtl_backward_fails_with_shared_activation_retaining_grad():
    """
    Tests that mtl_backward fails to fill a valid `.grad` when some tensor in the computation graph
    of the ``features`` parameter retains grad.
    """

    p0 = torch.tensor(1.0, requires_grad=True, device=DEVICE)
    p1 = torch.tensor(2.0, requires_grad=True, device=DEVICE)
    p2 = torch.tensor(3.0, requires_grad=True, device=DEVICE)

    a = 2 * p0
    a.retain_grad()
    features = 3 * a
    y1 = p1 * features
    y2 = p2 * features

    # mtl_backward itself doesn't raise the error, but it fills a.grad with a BatchedTensor
    mtl_backward(
        losses=[y1, y2],
        features=[features],
        aggregator=UPGrad(),
        tasks_params=[[p1], [p2]],
        shared_params=[p0],
    )

    with raises(RuntimeError):
        # Using such a BatchedTensor should result in an error
        _ = -a.grad


def test_mtl_backward_task_params_have_some_overlap():
    """Tests that mtl_backward works correctly when the tasks' parameters have some overlap."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor(2.0, requires_grad=True, device=DEVICE)
    p2 = torch.tensor(3.0, requires_grad=True, device=DEVICE)
    p12 = torch.tensor(4.0, requires_grad=True, device=DEVICE)

    r = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    y1 = r * p1 * p12
    y2 = r * p2 * p12

    aggregator = UPGrad()
    mtl_backward(losses=[y1, y2], features=[r], aggregator=aggregator, retain_graph=True)

    assert_close(p2.grad, r * p12)
    assert_close(p1.grad, r * p12)
    assert_close(p12.grad, r * p1 + r * p2)

    J = torch.tensor([[-p1 * p12, p1 * p12], [-p2 * p12, p2 * p12]], device=DEVICE)
    assert_close(p0.grad, aggregator(J))


def test_mtl_backward_task_params_are_the_same():
    """Tests that mtl_backward works correctly when the tasks have the same params."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor(2.0, requires_grad=True, device=DEVICE)

    r = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    y1 = r * p1
    y2 = r + p1

    aggregator = UPGrad()
    mtl_backward(losses=[y1, y2], features=[r], aggregator=aggregator, retain_graph=True)

    assert_close(p1.grad, r + 1)

    J = torch.tensor([[-p1, p1], [-1.0, 1.0]], device=DEVICE)
    assert_close(p0.grad, aggregator(J))


def test_mtl_backward_task_params_are_subset_of_other_task_params():
    """
    Tests that mtl_backward works correctly when one task's params are a subset of another task's
    params.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor(2.0, requires_grad=True, device=DEVICE)
    p2 = torch.tensor(3.0, requires_grad=True, device=DEVICE)

    r = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    y1 = r * p1
    y2 = y1 * p2

    aggregator = UPGrad()
    mtl_backward(losses=[y1, y2], features=[r], aggregator=aggregator, retain_graph=True)

    assert_close(p2.grad, y1)
    assert_close(p1.grad, p2 * r + r)

    J = torch.tensor([[-p1, p1], [-p1 * p2, p1 * p2]], device=DEVICE)
    assert_close(p0.grad, aggregator(J))


def test_mtl_backward_shared_params_overlap_with_tasks_params():
    """
    Tests that mtl_backward raises an error when the set of shared params overlaps with the set of
    task-specific params.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor(2.0, requires_grad=True, device=DEVICE)
    p2 = torch.tensor(3.0, requires_grad=True, device=DEVICE)

    r = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    y1 = r * p1
    y2 = p0.sum() * r * p2

    with raises(ValueError):
        mtl_backward(
            losses=[y1, y2],
            features=[r],
            aggregator=UPGrad(),
            tasks_params=[[p1], [p0, p2]],  # Problem: p0 is also shared
            shared_params=[p0],
            retain_graph=True,
        )


def test_mtl_backward_default_shared_params_overlap_with_default_tasks_params():
    """
    Tests that mtl_backward raises an error when the set of shared params obtained by default
    overlaps with the set of task-specific params obtained by default.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p1 = torch.tensor(2.0, requires_grad=True, device=DEVICE)
    p2 = torch.tensor(3.0, requires_grad=True, device=DEVICE)

    r = torch.tensor([-1.0, 1.0], device=DEVICE) @ p0
    y1 = r * p1
    y2 = p0.sum() * r * p2

    with raises(ValueError):
        mtl_backward(
            losses=[y1, y2],
            features=[r],
            aggregator=UPGrad(),
            retain_graph=True,
        )
