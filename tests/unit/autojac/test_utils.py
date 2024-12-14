import torch
from pytest import mark, raises
from torch.nn import Linear, MSELoss, ReLU, Sequential
from unit.conftest import DEVICE

from torchjd.autojac._utils import _get_leaf_tensors


def test_simple_get_leaf_tensors():
    """Tests that _get_leaf_tensors works correctly in a very simple setting."""

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    leaves = _get_leaf_tensors(tensors=[y1, y2], excluded=set())
    assert leaves == {a1, a2}


def test_get_leaf_tensors_excluded_1():
    """
    Tests that _get_leaf_tensors works correctly when some tensors are excluded from the search.

    Note that `a2` itself is not in `excluded`, but it is not accessible from `y1` or `y2` when `b2`
    is excluded from the graph traversal.
    """

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    b1 = (a1**2).sum()
    b2 = (a2**2).sum()

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + b2
    y2 = b1

    leaves = _get_leaf_tensors(tensors=[y1, y2], excluded={b1, b2})
    assert leaves == {a1}


def test_get_leaf_tensors_excluded_2():
    """
    Tests that _get_leaf_tensors works correctly when some tensors are excluded from the search.

    Even though `b1` and `b2`, that have `a1` and `a2` as descendants, are excluded, `y1` depends on
    `a1` and `a2` from another path, so these two tensors should be in the result.
    """

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    b1 = (a1**2).sum()
    b2 = (a2**2).sum()

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = b1

    leaves = _get_leaf_tensors(tensors=[y1, y2], excluded={b1, b2})
    assert leaves == {a1, a2}


def test_get_leaf_tensors_leaf_not_requiring_grad():
    """
    Tests that _get_leaf_tensors does not include tensors that do not require grad in its results.
    """

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=False, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum() + a2.norm()

    leaves = _get_leaf_tensors(tensors=[y1, y2], excluded=set())
    assert leaves == {a1}


def test_get_leaf_tensors_model():
    """
    Tests that _get_leaf_tensors works correctly when the autograd graph is generated by a simple
    sequential model.
    """

    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
    loss_fn = MSELoss(reduction="none")

    y_hat = model(x)
    losses = loss_fn(y_hat, y)

    leaves = _get_leaf_tensors(tensors=[losses], excluded=set())
    assert leaves == set(model.parameters())


def test_get_leaf_tensors_model_excluded_2():
    """
    Tests that _get_leaf_tensors works correctly when the autograd graph is generated by a simple
    sequential model, and some intermediate values are excluded.
    """

    x = torch.randn(16, 10)
    z = torch.randn(16, 1)

    model1 = Sequential(Linear(10, 5), ReLU())
    model2 = Linear(5, 1)
    loss_fn = MSELoss(reduction="none")

    y = model1(x)
    z_hat = model2(y)
    losses = loss_fn(z_hat, z)

    leaves = _get_leaf_tensors(tensors=[losses], excluded={y})
    assert leaves == set(model2.parameters())


def test_get_leaf_tensors_single_root():
    """Tests that _get_leaf_tensors returns no leaves when roots is the empty set."""

    p = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    y = p * 2

    leaves = _get_leaf_tensors(tensors=[y], excluded=set())
    assert leaves == {p}


def test_get_leaf_tensors_empty_roots():
    """Tests that _get_leaf_tensors returns no leaves when roots is the empty set."""

    leaves = _get_leaf_tensors(tensors=[], excluded=set())
    assert leaves == set()


def test_get_leaf_tensors_excluded_root():
    """Tests that _get_leaf_tensors correctly excludes the root."""

    a1 = torch.tensor([1.0, 2.0], requires_grad=True, device=DEVICE)
    a2 = torch.tensor([3.0, 4.0], requires_grad=True, device=DEVICE)

    y1 = torch.tensor([-1.0, 1.0], device=DEVICE) @ a1 + a2.sum()
    y2 = (a1**2).sum()

    leaves = _get_leaf_tensors(tensors=[y1, y2], excluded={y1})
    assert leaves == {a1}


@mark.parametrize("depth", [100, 1000, 10000])
def test_get_leaf_tensors_deep(depth: int):
    """Tests that _get_leaf_tensors works when the graph is very deep."""

    one = torch.tensor(1.0, requires_grad=True, device=DEVICE)
    sum_ = torch.tensor(0.0, requires_grad=False, device=DEVICE)
    for i in range(depth):
        sum_ = sum_ + one

    leaves = _get_leaf_tensors(tensors=[sum_], excluded=set())
    assert leaves == {one}


def test_get_leaf_tensors_leaf():
    """Tests that _get_leaf_tensors raises an error some of the provided tensors are leaves."""

    a = torch.tensor(1.0, requires_grad=True, device=DEVICE)
    with raises(ValueError):
        _ = _get_leaf_tensors(tensors=[a], excluded=set())


def test_get_leaf_tensors_tensor_not_requiring_grad():
    """
    Tests that _get_leaf_tensors raises an error some of the provided tensors do not require grad.
    """

    a = torch.tensor(1.0, requires_grad=False, device=DEVICE) * 2
    with raises(ValueError):
        _ = _get_leaf_tensors(tensors=[a], excluded=set())


def test_get_leaf_tensors_excluded_leaf():
    """Tests that _get_leaf_tensors raises an error some of the excluded tensors are leaves."""

    a = torch.tensor(1.0, requires_grad=True, device=DEVICE) * 2
    b = torch.tensor(2.0, requires_grad=True, device=DEVICE)
    with raises(ValueError):
        _ = _get_leaf_tensors(tensors=[a], excluded={b})


def test_get_leaf_tensors_excluded_not_requiring_grad():
    """
    Tests that _get_leaf_tensors raises an error some of the excluded tensors do not require grad.
    """

    a = torch.tensor(1.0, requires_grad=True, device=DEVICE) * 2
    b = torch.tensor(2.0, requires_grad=False, device=DEVICE) * 2
    with raises(ValueError):
        _ = _get_leaf_tensors(tensors=[a], excluded={b})
