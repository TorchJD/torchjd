import torch
from torch.nn import Linear, MSELoss, ReLU, Sequential
from unit.conftest import DEVICE

from torchjd.autojac._utils import _get_leaves_of_autograd_graph


def test_simple_get_leaves_of_autograd_graph():
    """Tests that _get_leaves_of_autograd_graph works correctly in a very simple setting."""

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    expected = {p1, p2}
    leaves = _get_leaves_of_autograd_graph(roots=[y1, y2], excluded=set())

    assert leaves == expected


def test_simple_get_leaves_of_autograd_graph_excluded_1():
    """
    Tests that _get_leaves_of_autograd_graph works correctly when some tensors are excluded from the
    search.

    Note that `p2` itself is not in `excluded`, but it is not accessible from `y1` or `y2` when `x2`
    is excluded from the graph traversal.
    """

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    x1 = (p1**2).sum()
    x2 = (p2**2).sum()

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + x2
    y2 = x1

    expected = {p1}
    leaves = _get_leaves_of_autograd_graph(roots=[y1, y2], excluded={x1, x2})

    assert leaves == expected


def test_simple_get_leaves_of_autograd_graph_excluded_2():
    """
    Tests that _get_leaves_of_autograd_graph works correctly when some tensors are excluded from the
    search.

    Even though `x1` and `x2`, that have `p1` and `p2` as descendants, are excluded, `y1` depends on
    `p1` and `p2` from another path, so these two tensors should be in the result.
    """

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    x1 = (p1**2).sum()
    x2 = (p2**2).sum()

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = x1

    expected = {p1, p2}
    leaves = _get_leaves_of_autograd_graph(roots=[y1, y2], excluded={x1, x2})

    assert leaves == expected


def test_simple_get_leaves_of_autograd_graph_excluded_3():
    """
    Tests that _get_leaves_of_autograd_graph works correctly when some tensors are excluded from the
    search.

    In this case, one of the leaves itself is excluded.
    """

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)
    p3 = torch.tensor([5.0, 6.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm() + p3.sum()

    expected = {p1, p2}
    leaves = _get_leaves_of_autograd_graph(roots=[y1, y2], excluded={p3})

    assert leaves == expected


def test_simple_get_leaves_of_autograd_graph_with_leaf_not_requiring_grad():
    """
    Tests that _get_leaves_of_autograd_graph does not include tensors that do not require grad in
    its results.
    """

    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    p2 = torch.tensor([3.0, 4.0], requires_grad=False, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ p1 + p2.sum()
    y2 = (p1**2).sum() + p2.norm()

    expected = {p1}
    leaves = _get_leaves_of_autograd_graph(roots=[y1, y2], excluded=set())

    assert leaves == expected


def test_simple_get_leaves_of_autograd_graph_with_model():
    """
    Tests that _get_leaves_of_autograd_graph works correctly when the autograd graph is generated by
    a simple sequential model.
    """

    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    loss_fn = MSELoss(reduction="none")

    y_hat = model(x)
    losses = loss_fn(y_hat, y)

    expected = set(model.parameters())
    leaves = _get_leaves_of_autograd_graph(roots=[losses], excluded=set())

    assert leaves == expected


def test_simple_get_leaves_of_autograd_graph_with_model_excluded_1():
    """
    Tests that _get_leaves_of_autograd_graph works correctly when the autograd graph is generated by
    a simple sequential model, and some of the model's parameters are excluded.
    """

    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    loss_fn = MSELoss(reduction="none")

    y_hat = model(x)
    losses = loss_fn(y_hat, y)

    expected = set(model[2].parameters())
    leaves = _get_leaves_of_autograd_graph(roots=[losses], excluded=set(model[0].parameters()))

    assert leaves == expected


def test_simple_get_leaves_of_autograd_graph_with_model_excluded_2():
    """
    Tests that _get_leaves_of_autograd_graph works correctly when the autograd graph is generated by
    a simple sequential model, and some intermediate values are excluded.
    """

    x = torch.randn(16, 10)
    z = torch.randn(16, 1)

    model1 = Sequential(Linear(10, 5), ReLU())
    model2 = Linear(5, 1)
    loss_fn = MSELoss(reduction="none")

    y = model1(x)
    z_hat = model2(y)
    losses = loss_fn(z_hat, z)

    expected = set(model2.parameters())
    leaves = _get_leaves_of_autograd_graph(roots=[losses], excluded={y})

    assert leaves == expected