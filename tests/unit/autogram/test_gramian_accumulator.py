from pytest import mark
from torch.testing import assert_close
from utils.tensors import randn_, zeros_

from torchjd._autogram._gramian_accumulator import GramianAccumulator


@mark.parametrize(
    ["shapes", "number_of_jacobians"],
    [
        ([[3, 4, 5], [7, 5]], [3, 7]),
        ([[3], [7, 5, 8], [2, 3]], [0, 7, 1]),
    ],
)
def test_adding_jacobians_one_by_one(shapes, number_of_jacobians):
    batch_size = 10
    gramian_accumulator = GramianAccumulator()

    keys = [randn_(shape) for shape in shapes]
    for key, n in zip(keys, number_of_jacobians):
        gramian_accumulator.track_parameter_paths([key] * n)

    expected_gramian = zeros_([batch_size, batch_size])

    for key, shape, n in zip(keys, shapes, number_of_jacobians):
        batched_shape = [batch_size] + shape
        cumulated_jacobian = zeros_(batched_shape)
        for i in range(n):
            jacobian = randn_(batched_shape)
            gramian_accumulator.accumulate_path_jacobians({key: jacobian})
            cumulated_jacobian += jacobian
        jacobian_matrix = cumulated_jacobian.reshape([batch_size, -1])
        expected_gramian.addmm_(jacobian_matrix, jacobian_matrix.T)

    gramian = gramian_accumulator.gramian
    assert_close(gramian, expected_gramian)
    assert_close(gramian, expected_gramian)


@mark.parametrize(
    ["shapes"],
    [
        ([[3, 4, 5], [7, 5]],),
        ([[3], [7, 5, 8], [2, 3]],),
    ],
)
def test_adding_jacobians_lots_by_lots(shapes):
    number_of_jacobians = 4
    batch_size = 10
    gramian_accumulator = GramianAccumulator()

    keys = [randn_(shape) for shape in shapes]
    for i in range(number_of_jacobians):
        gramian_accumulator.track_parameter_paths(keys)

    expected_gramian = zeros_([batch_size, batch_size])

    cumulated_jacobians = {key: zeros_([batch_size] + shape) for key, shape in zip(keys, shapes)}
    for i in range(number_of_jacobians):
        jacobians = {key: randn_([batch_size] + shape) for key, shape in zip(keys, shapes)}
        gramian_accumulator.accumulate_path_jacobians(jacobians)
        for key, jacobian in jacobians.items():
            cumulated_jacobians[key] += jacobian
    for cumulated_jacobian in cumulated_jacobians.values():
        jacobian_matrix = cumulated_jacobian.reshape([batch_size, -1])
        expected_gramian.addmm_(jacobian_matrix, jacobian_matrix.T)

    gramian = gramian_accumulator.gramian
    assert_close(gramian, expected_gramian, rtol=5e-06, atol=2e-05)


def test_returns_none_if_no_jacobian_were_provided():
    gramian_accumulator = GramianAccumulator()
    assert gramian_accumulator.gramian is None


@mark.parametrize(
    ["shapes", "number_of_jacobians"],
    [
        ([[3, 4, 5], [7, 5]], [3, 7]),
        ([[3], [7, 5, 8], [2, 3]], [0, 7, 1]),
    ],
)
def test_internal_dicts_are_cleaned(shapes, number_of_jacobians):
    batch_size = 10
    gramian_accumulator = GramianAccumulator()

    keys = [randn_(shape) for shape in shapes]
    for key, n in zip(keys, number_of_jacobians):
        gramian_accumulator.track_parameter_paths([key] * n)

    for key, shape, n in zip(keys, shapes, number_of_jacobians):
        batched_shape = [batch_size] + shape
        for i in range(n):
            jacobian = randn_(batched_shape)
            gramian_accumulator.accumulate_path_jacobians({key: jacobian})
        assert key not in gramian_accumulator._summed_jacobians.keys()
        assert key not in gramian_accumulator._path_counter.keys()
