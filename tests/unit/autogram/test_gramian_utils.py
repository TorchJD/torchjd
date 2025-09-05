from math import prod

import torch
from pytest import mark
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import rand_

from torchjd.autogram._gramian_utils import movedim_gramian, reshape_gramian


def compute_quadratic_form(generalized_gramian: Tensor, x: Tensor) -> Tensor:
    """
    Compute the quadratic form x^T G x when the provided generalized Gramian and x may have multiple
    dimensions.
    """
    indices = list(range(x.ndim))
    linear_form = torch.tensordot(x, generalized_gramian, dims=(indices, indices))
    return torch.tensordot(linear_form, x, dims=(indices[::-1], indices))


@mark.parametrize(
    "shape",
    [
        [50, 2, 2, 3],
        [60, 3, 2, 5],
        [30, 6, 7],
        [4, 3, 1],
        [4, 1, 1],
        [1, 1, 1],
        [4, 1],
        [4],
        [1, 1],
        [1],
    ],
)
def test_quadratic_form_invariance_to_reshape(shape: list[int]):
    """
    When reshaping a Gramian, we expect it to represent the same quadratic form that now applies to
    reshaped inputs. So the mapping x -> x^T G x commutes with reshaping x, G and then computing the
    corresponding quadratic form.
    """

    flat_dim = prod(shape[1:])
    iterations = 20

    matrix = rand_([flat_dim, shape[0]])
    gramian = matrix @ matrix.T
    reshaped_gramian = reshape_gramian(gramian, shape[1:])

    for _ in range(iterations):
        vector = rand_([flat_dim])
        reshaped_vector = vector.reshape(shape[1:])

        quadratic_form = vector @ gramian @ vector
        reshaped_quadratic_form = compute_quadratic_form(reshaped_gramian, reshaped_vector)

        assert_close(reshaped_quadratic_form, quadratic_form)


@mark.parametrize(
    ["shape", "source", "destination"],
    [
        ([50, 2, 2, 3], [0, 2], [1, 0]),
        ([60, 3, 2, 5], [1], [2]),
        ([30, 6, 7], [0, 1], [1, 0]),
    ],
)
def test_quadratic_form_invariance_to_movedim(
    shape: list[int], source: list[int], destination: list[int]
):
    """
    When moving dims on a Gramian, we expect it to represent the same quadratic form that now
    applies to inputs with moved dims. So the mapping x -> x^T G x commutes with moving dims x, G
    and then computing the quadratic form with those.
    """

    flat_dim = prod(shape[1:])
    iterations = 20

    matrix = rand_([flat_dim, shape[0]])
    gramian = reshape_gramian(matrix @ matrix.T, shape[1:])
    moved_gramian = movedim_gramian(gramian, source, destination)

    for _ in range(iterations):
        vector = rand_(shape[1:])
        moved_vector = vector.movedim(source, destination)

        quadratic_form = compute_quadratic_form(gramian, vector)
        moved_quadratic_form = compute_quadratic_form(moved_gramian, moved_vector)

        assert_close(moved_quadratic_form, quadratic_form)
