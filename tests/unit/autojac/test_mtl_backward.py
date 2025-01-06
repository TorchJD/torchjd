import torch
from pytest import mark, raises
from torch.testing import assert_close

from torchjd import mtl_backward
from torchjd.aggregation import MGDA, Aggregator, Mean, Random, UPGrad


@mark.parametrize("aggregator", [Mean(), UPGrad(), MGDA(), Random()])
def test_various_aggregators(aggregator: Aggregator):
    """Tests that mtl_backward works for various aggregators."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    mtl_backward(losses=[y1, y2], features=[f1, f2], aggregator=aggregator)

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@mark.parametrize("aggregator", [Mean(), UPGrad()])
@mark.parametrize("shape", [(1, 3), (2, 3), (2, 6), (5, 8), (20, 55)])
@mark.parametrize("manually_specify_shared_params", [True, False])
@mark.parametrize("manually_specify_tasks_params", [True, False])
@mark.parametrize("chunk_size", [1, 2, None])
def test_value_is_correct(
    aggregator: Aggregator,
    shape: tuple[int, int],
    manually_specify_shared_params: bool,
    manually_specify_tasks_params: bool,
    chunk_size: int | None,
):
    """
    Tests that the .grad value filled by mtl_backward is correct in a simple example of
    matrix-vector product for three tasks whose loss are given by a simple inner product of the
    shared features with the task parameter.

    This test should work with or without manually specifying the parameters.
    """

    p0 = torch.randn([shape[1]], requires_grad=True)
    p1 = torch.randn(shape[0], requires_grad=True)
    p2 = torch.randn(shape[0], requires_grad=True)
    p3 = torch.randn(shape[0], requires_grad=True)

    J = torch.randn(shape)
    f = J @ p0
    y1 = p1 @ f
    y2 = p2 @ f
    y3 = p3 @ f

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
        features=f,
        aggregator=aggregator,
        tasks_params=tasks_params,
        shared_params=shared_params,
        parallel_chunk_size=chunk_size,
    )

    assert_close(p1.grad, f)
    assert_close(p2.grad, f)
    assert_close(p3.grad, f)

    expected_jacobian = torch.stack((p1, p2, p3)) @ J
    expected_aggregation = aggregator(expected_jacobian)

    assert_close(p0.grad, expected_aggregation)


def test_empty_tasks_fails():
    """Tests that mtl_backward raises an error when called with an empty list of tasks."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()

    with raises(ValueError):
        mtl_backward(losses=[], features=[f1, f2], aggregator=UPGrad())


def test_single_task():
    """Tests that mtl_backward works correctly with a single task."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]

    mtl_backward(losses=[y1], features=[f1, f2], aggregator=UPGrad())

    for p in [p0, p1]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


def test_incoherent_task_number_fails():
    """
    Tests that mtl_backward raises an error when called with the number of tasks losses different
    from the number of tasks parameters.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        mtl_backward(
            losses=[y1, y2],
            features=[f1, f2],
            aggregator=UPGrad(),
            tasks_params=[[p1]],  # Wrong
            shared_params=[p0],
        )
    with raises(ValueError):
        mtl_backward(
            losses=[y1],  # Wrong
            features=[f1, f2],
            aggregator=UPGrad(),
            tasks_params=[[p1], [p2]],
            shared_params=[p0],
        )


def test_empty_params():
    """Tests that mtl_backward does not fill the .grad values if no parameter is specified."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    mtl_backward(
        losses=[y1, y2],
        features=[f1, f2],
        aggregator=UPGrad(),
        tasks_params=[[], []],
        shared_params=[],
    )

    for p in [p0, p1, p2]:
        assert p.grad is None


def test_multiple_params_per_task():
    """Tests that mtl_backward works correctly when the tasks each have several parameters."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1_a = torch.tensor(1.0, requires_grad=True)
    p1_b = torch.tensor([2.0, 3.0], requires_grad=True)
    p1_c = torch.tensor([[4.0, 5.0], [6.0, 7.0]], requires_grad=True)
    p2_a = torch.tensor(8.0, requires_grad=True)
    p2_b = torch.tensor([9.0, 10.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1_a + (f2 * p1_b).sum() + (f1 * p1_c).sum()
    y2 = f1 * p2_a * (f2 * p2_b).sum()

    mtl_backward(losses=[y1, y2], features=[f1, f2], aggregator=UPGrad())

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
def test_various_shared_params(shared_params_shapes: list[tuple[int]]):
    """Tests that mtl_backward works correctly with various kinds of shared_params."""

    shared_params = [torch.rand(shape, requires_grad=True) for shape in shared_params_shapes]
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    features = [shared_param.sum(dim=-1) for shared_param in shared_params]
    y1 = torch.stack([f.sum() for f in features]).sum()
    y2 = torch.stack([f.sum() ** 2 for f in features]).sum()

    mtl_backward(
        losses=[y1, y2],
        features=features,
        aggregator=UPGrad(),
        tasks_params=[[p1], [p2]],  # Enforce differentiation w.r.t. params that haven't been used
        shared_params=shared_params,
    )

    for p in [*shared_params, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


def test_partial_params():
    """
    Tests that mtl_backward fills the right .grad values when only a subset of the parameters are
    specified as inputs.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    mtl_backward(
        losses=[y1, y2],
        features=[f1, f2],
        aggregator=Mean(),
        tasks_params=[[p1], []],
        shared_params=[p0],
    )

    assert (p0.grad is not None) and (p0.shape == p0.grad.shape)
    assert (p1.grad is not None) and (p1.shape == p1.grad.shape)
    assert p2.grad is None


def test_empty_features_fails():
    """Tests that mtl_backward expectedly raises an error when no there is no feature."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

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
def test_various_single_features(shape: tuple[int, ...]):
    """Tests that mtl_backward works correctly with various kinds of feature tensors."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([3.0, 4.0], requires_grad=True)
    p2 = torch.tensor([5.0, 6.0], requires_grad=True)

    f = torch.rand(shape) @ p0

    y1 = (f * p1[0]).sum() + (f * p1[1]).sum()
    y2 = (f * p2[0]).sum() * (f * p2[1]).sum()

    mtl_backward(losses=[y1, y2], features=f, aggregator=UPGrad())

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
def test_various_feature_lists(shapes: list[tuple[int]]):
    """Tests that mtl_backward works correctly with various kinds of feature lists."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.arange(len(shapes), dtype=torch.float32, requires_grad=True)
    p2 = torch.tensor(5.0, requires_grad=True)

    features = [torch.rand(shape) @ p0 for shape in shapes]

    y1 = sum([(f * p).sum() for f, p in zip(features, p1)])
    y2 = (features[0] * p2).sum()

    mtl_backward(losses=[y1, y2], features=features, aggregator=UPGrad())

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


def test_non_scalar_loss_fails():
    """Tests that mtl_backward raises an error when used with a non-scalar loss."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = torch.stack([f1 * p1[0], f2 * p1[1]])  # Non-scalar
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        mtl_backward(losses=[y1, y2], features=[f1, f2], aggregator=UPGrad())


@mark.parametrize("chunk_size", [None, 1, 2, 4])
def test_various_valid_chunk_sizes(chunk_size):
    """Tests that mtl_backward works for various valid values of parallel_chunk_size."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    mtl_backward(
        losses=[y1, y2],
        features=[f1, f2],
        aggregator=UPGrad(),
        parallel_chunk_size=chunk_size,
    )

    for p in [p0, p1, p2]:
        assert (p.grad is not None) and (p.shape == p.grad.shape)


@mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_fails(chunk_size: int):
    """Tests that mtl_backward raises an error when using invalid chunk sizes."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)

    f1 = torch.tensor([-1.0, 1.0]) @ p0
    f2 = (p0**2).sum() + p0.norm()
    y1 = f1 * p1[0] + f2 * p1[1]
    y2 = f1 * p2[0] + f2 * p2[1]

    with raises(ValueError):
        mtl_backward(
            losses=[y1, y2],
            features=[f1, f2],
            aggregator=UPGrad(),
            parallel_chunk_size=chunk_size,
        )


def test_shared_param_retaining_grad_fails():
    """
    Tests that mtl_backward raises an error when some shared param in the computation graph of the
    ``features`` parameter retains grad and vmap has to be used.
    """

    p0 = torch.tensor(1.0, requires_grad=True)
    p1 = torch.tensor(2.0, requires_grad=True)
    p2 = torch.tensor(3.0, requires_grad=True)

    a = 2 * p0
    a.retain_grad()
    f = 3 * a
    y1 = p1 * f
    y2 = p2 * f

    with raises(RuntimeError):
        mtl_backward(
            losses=[y1, y2],
            features=[f],
            aggregator=UPGrad(),
            tasks_params=[[p1], [p2]],
            shared_params=[a, p0],
        )


def test_shared_activation_retaining_grad_fails():
    """
    Tests that mtl_backward fails to fill a valid `.grad` when some tensor in the computation graph
    of the ``features`` parameter retains grad and vmap has to be used.
    """

    p0 = torch.tensor(1.0, requires_grad=True)
    p1 = torch.tensor(2.0, requires_grad=True)
    p2 = torch.tensor(3.0, requires_grad=True)

    a = 2 * p0
    a.retain_grad()
    f = 3 * a
    y1 = p1 * f
    y2 = p2 * f

    # mtl_backward itself doesn't raise the error, but it fills a.grad with a BatchedTensor
    mtl_backward(
        losses=[y1, y2],
        features=[f],
        aggregator=UPGrad(),
        tasks_params=[[p1], [p2]],
        shared_params=[p0],
    )

    with raises(RuntimeError):
        # Using such a BatchedTensor should result in an error
        _ = -a.grad


def test_tasks_params_overlap():
    """Tests that mtl_backward works correctly when the tasks' parameters have some overlap."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor(2.0, requires_grad=True)
    p2 = torch.tensor(3.0, requires_grad=True)
    p12 = torch.tensor(4.0, requires_grad=True)

    f = torch.tensor([-1.0, 1.0]) @ p0
    y1 = f * p1 * p12
    y2 = f * p2 * p12

    aggregator = UPGrad()
    mtl_backward(losses=[y1, y2], features=[f], aggregator=aggregator)

    assert_close(p2.grad, f * p12)
    assert_close(p1.grad, f * p12)
    assert_close(p12.grad, f * p1 + f * p2)

    J = torch.tensor([[-p1 * p12, p1 * p12], [-p2 * p12, p2 * p12]])
    assert_close(p0.grad, aggregator(J))


def test_tasks_params_are_the_same():
    """Tests that mtl_backward works correctly when the tasks have the same params."""

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor(2.0, requires_grad=True)

    f = torch.tensor([-1.0, 1.0]) @ p0
    y1 = f * p1
    y2 = f + p1

    aggregator = UPGrad()
    mtl_backward(losses=[y1, y2], features=[f], aggregator=aggregator)

    assert_close(p1.grad, f + 1)

    J = torch.tensor([[-p1, p1], [-1.0, 1.0]])
    assert_close(p0.grad, aggregator(J))


def test_task_params_is_subset_of_other_task_params():
    """
    Tests that mtl_backward works correctly when one task's params is a subset of another task's
    params.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor(2.0, requires_grad=True)
    p2 = torch.tensor(3.0, requires_grad=True)

    f = torch.tensor([-1.0, 1.0]) @ p0
    y1 = f * p1
    y2 = y1 * p2

    aggregator = UPGrad()
    mtl_backward(losses=[y1, y2], features=[f], aggregator=aggregator, retain_graph=True)

    assert_close(p2.grad, y1)
    assert_close(p1.grad, p2 * f + f)

    J = torch.tensor([[-p1, p1], [-p1 * p2, p1 * p2]])
    assert_close(p0.grad, aggregator(J))


def test_shared_params_overlapping_with_tasks_params_fails():
    """
    Tests that mtl_backward raises an error when the set of shared params overlaps with the set of
    task-specific params.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor(2.0, requires_grad=True)
    p2 = torch.tensor(3.0, requires_grad=True)

    f = torch.tensor([-1.0, 1.0]) @ p0
    y1 = f * p1
    y2 = p0.sum() * f * p2

    with raises(ValueError):
        mtl_backward(
            losses=[y1, y2],
            features=[f],
            aggregator=UPGrad(),
            tasks_params=[[p1], [p0, p2]],  # Problem: p0 is also shared
            shared_params=[p0],
        )


def test_default_shared_params_overlapping_with_default_tasks_params_fails():
    """
    Tests that mtl_backward raises an error when the set of shared params obtained by default
    overlaps with the set of task-specific params obtained by default.
    """

    p0 = torch.tensor([1.0, 2.0], requires_grad=True)
    p1 = torch.tensor(2.0, requires_grad=True)
    p2 = torch.tensor(3.0, requires_grad=True)

    f = torch.tensor([-1.0, 1.0]) @ p0
    y1 = f * p1
    y2 = p0.sum() * f * p2

    with raises(ValueError):
        mtl_backward(
            losses=[y1, y2],
            features=[f],
            aggregator=UPGrad(),
        )
