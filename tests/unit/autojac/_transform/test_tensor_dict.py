from contextlib import nullcontext as does_not_raise

import torch
from pytest import mark, raises
from torch import Tensor
from unit._utils import ExceptionContext

from torchjd.autojac._transform import (
    EmptyTensorDict,
    Gradients,
    GradientVectors,
    JacobianMatrices,
    Jacobians,
    TensorDict,
)
from torchjd.autojac._transform.tensor_dict import _least_common_ancestor

_key_shapes = [[], [1], [2, 3]]


@mark.parametrize(
    ["value_shapes", "expectation"],
    [
        ([[], [1], [2, 3]], does_not_raise()),
        ([[1], [1], [2, 3]], raises(ValueError)),  # 1 extra dimension
        ([[], [1], [6]], raises(ValueError)),  # 1 missing dimension
        ([[], [1], [2, 4]], raises(ValueError)),  # Wrong number of elements
    ],
)
def test_gradients(value_shapes: list[list[int]], expectation: ExceptionContext):
    """Tests that the Gradients class checks properly its inputs."""

    _assert_class_checks_properly(Gradients, value_shapes, expectation)


@mark.parametrize(
    ["value_shapes", "expectation"],
    [
        ([[5], [5, 1], [5, 2, 3]], does_not_raise()),
        ([[5], [5, 1], [6, 2, 3]], raises(ValueError)),  # Different first dimension
        ([[5, 1], [5, 1], [5, 2, 3]], raises(ValueError)),  # 1 extra dimension
        ([[5], [5, 1], [5, 6]], raises(ValueError)),  # 1 missing dimension
        ([[5], [5, 1], [5, 2, 4]], raises(ValueError)),  # Wrong number of elements
    ],
)
def test_jacobians(value_shapes: list[list[int]], expectation: ExceptionContext):
    """Tests that the Jacobians class checks properly its inputs."""

    _assert_class_checks_properly(Jacobians, value_shapes, expectation)


@mark.parametrize(
    ["value_shapes", "expectation"],
    [
        ([[1], [1], [6]], does_not_raise()),
        ([[], [1], [6]], raises(ValueError)),  # Not a vector
        ([[1], [1], [2, 3]], raises(ValueError)),  # Not a vector
        ([[2], [1], [6]], raises(ValueError)),  # Wrong number of elements
    ],
)
def test_gradient_vectors(value_shapes: list[list[int]], expectation: ExceptionContext):
    """Tests that the GradientVectors class checks properly its inputs."""

    _assert_class_checks_properly(GradientVectors, value_shapes, expectation)


@mark.parametrize(
    ["value_shapes", "expectation"],
    [
        ([[5, 1], [5, 1], [5, 6]], does_not_raise()),
        ([[5, 1], [5, 1], [6, 6]], raises(ValueError)),  # Different first dimension
        ([[5], [5, 1], [5, 6]], raises(ValueError)),  # Not a matrix
        ([[5, 1], [5, 1], [5, 2, 3]], raises(ValueError)),  # Not a matrix
        ([[5, 2], [5, 1], [5, 6]], raises(ValueError)),  # Wrong number of elements
    ],
)
def test_jacobian_matrices(value_shapes: list[list[int]], expectation: ExceptionContext):
    """Tests that the JacobianMatrices class checks properly its inputs."""

    _assert_class_checks_properly(JacobianMatrices, value_shapes, expectation)


@mark.parametrize(
    ["first", "second", "result"],
    [
        (EmptyTensorDict, EmptyTensorDict, EmptyTensorDict),
        (EmptyTensorDict, Jacobians, Jacobians),
        (Jacobians, EmptyTensorDict, Jacobians),
        (Jacobians, Jacobians, Jacobians),
        (EmptyTensorDict, Gradients, Gradients),
        (EmptyTensorDict, GradientVectors, GradientVectors),
        (EmptyTensorDict, JacobianMatrices, JacobianMatrices),
        (GradientVectors, JacobianMatrices, TensorDict),
    ],
)
def test_least_common_ancestor(
    first: type[TensorDict], second: type[TensorDict], result: type[TensorDict]
):
    assert _least_common_ancestor(first, second) == result


def _assert_class_checks_properly(
    class_: type[TensorDict], value_shapes: list[list[int]], expectation: ExceptionContext
):
    tensor_mapping = _make_tensor_dict(value_shapes)

    with expectation:
        class_(tensor_mapping)


def _make_tensor_dict(value_shapes: list[list[int]]) -> dict[Tensor, Tensor]:
    return {torch.zeros(key): torch.zeros(value) for key, value in zip(_key_shapes, value_shapes)}
