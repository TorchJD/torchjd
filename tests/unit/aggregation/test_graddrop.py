from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from unit._utils import ExceptionContext

from torchjd.aggregation import GradDrop

from ._property_testers import ExpectedStructureProperty


@mark.parametrize("aggregator", [GradDrop()])
class TestGradDrop(ExpectedStructureProperty):
    pass


@mark.parametrize(
    ["leak_shape", "expectation"],
    [
        ([], raises(ValueError)),
        ([0], does_not_raise()),
        ([1], does_not_raise()),
        ([10], does_not_raise()),
        ([0, 0], raises(ValueError)),
        ([0, 1], raises(ValueError)),
        ([1, 1], raises(ValueError)),
        ([1, 1, 1], raises(ValueError)),
        ([1, 1, 1, 1], raises(ValueError)),
        ([1, 1, 1, 1, 1], raises(ValueError)),
    ],
)
def test_leak_shape_check(leak_shape: list[int], expectation: ExceptionContext):
    leak = torch.ones(leak_shape)
    with expectation:
        _ = GradDrop(leak=leak)


@mark.parametrize(
    ["leak_shape", "n_rows", "expectation"],
    [
        ([0], 0, does_not_raise()),
        ([1], 1, does_not_raise()),
        ([5], 5, does_not_raise()),
        ([0], 1, raises(ValueError)),
        ([1], 0, raises(ValueError)),
        ([4], 5, raises(ValueError)),
        ([5], 4, raises(ValueError)),
    ],
)
def test_matrix_shape_check(leak_shape: list[int], n_rows: int, expectation: ExceptionContext):
    matrix = torch.ones([n_rows, 5])
    leak = torch.ones(leak_shape)
    aggregator = GradDrop(leak=leak)

    with expectation:
        _ = aggregator(matrix)


def test_representations():
    A = GradDrop(leak=torch.tensor([0.0, 1.0], device="cpu"))
    assert repr(A) == "GradDrop(leak=tensor([0., 1.]))"
    assert str(A) == "GradDrop([0., 1.])"

    A = GradDrop()
    assert repr(A) == "GradDrop(leak=None)"
    assert str(A) == "GradDrop"
