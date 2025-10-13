from pytest import mark
from torch import nn
from torch.testing import assert_close
from utils.tensors import randn_, zeros_

from torchjd.autogram._gramian_accumulator import GramianAccumulator


class FakeModule(nn.Module):
    pass


@mark.parametrize(
    ["shapes", "number_of_jacobians"],
    [
        ([[3, 4, 5], [7, 5]], [3, 7]),
        ([[3], [7, 5, 8], [2, 3]], [0, 7, 1]),
    ],
)
def test_adding_jacobians_one_by_one(shapes: list[list[int]], number_of_jacobians: list[int]):
    batch_size = 10
    gramian_accumulator = GramianAccumulator()

    keys = [FakeModule() for _ in shapes]
    for key, n in zip(keys, number_of_jacobians):
        for _ in range(n):
            gramian_accumulator.track_module_paths(key)

    expected_gramian = zeros_([batch_size, batch_size])

    for key, shape, n in zip(keys, shapes, number_of_jacobians):
        batched_shape = [batch_size] + shape
        cumulated_jacobian = zeros_(batched_shape)
        for i in range(n):
            jacobian = randn_(batched_shape)
            gramian_accumulator.accumulate_path_jacobians(key, [jacobian])
            cumulated_jacobian += jacobian
        jacobian_matrix = cumulated_jacobian.reshape([batch_size, -1])
        expected_gramian.addmm_(jacobian_matrix, jacobian_matrix.T)

    gramian = gramian_accumulator.gramian
    assert_close(gramian, expected_gramian, rtol=5e-06, atol=2e-05)


@mark.parametrize(
    "shapes",
    [
        [[3, 4, 5], [7, 5]],
        [[3], [7, 5, 8], [2, 3]],
    ],
)
def test_adding_jacobians_lots_by_lots(shapes: list[list[int]]):
    number_of_jacobians = 4
    batch_size = 10
    gramian_accumulator = GramianAccumulator()

    keys = [FakeModule() for _ in shapes]
    for key in keys:
        for i in range(number_of_jacobians):
            gramian_accumulator.track_module_paths(key)

    expected_gramian = zeros_([batch_size, batch_size])

    cumulated_jacobians = {
        key: [zeros_([batch_size] + shape)] * number_of_jacobians
        for key, shape in zip(keys, shapes)
    }
    for i in range(number_of_jacobians):
        jacobian_dict = {
            key: [randn_([batch_size] + shape) for _ in range(number_of_jacobians)]
            for key, shape in zip(keys, shapes)
        }
        for key, jacobians in jacobian_dict.items():
            gramian_accumulator.accumulate_path_jacobians(key, jacobian_dict[key])
            cumulated_jacobians[key] = [a + b for a, b in zip(jacobians, cumulated_jacobians[key])]
    for cumulated_jacobian in cumulated_jacobians.values():
        for jacobian in cumulated_jacobian:
            jacobian_matrix = jacobian.reshape([batch_size, -1])
            expected_gramian.addmm_(jacobian_matrix, jacobian_matrix.T)

    gramian = gramian_accumulator.gramian
    assert_close(gramian, expected_gramian)


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
def test_internal_dicts_are_cleaned(shapes: list[list[int]], number_of_jacobians: list[int]):
    batch_size = 10
    gramian_accumulator = GramianAccumulator()

    keys = [FakeModule() for shape in shapes]
    for key, n in zip(keys, number_of_jacobians):
        for _ in range(n):
            gramian_accumulator.track_module_paths(key)

    for key, shape, n in zip(keys, shapes, number_of_jacobians):
        batched_shape = [batch_size] + shape
        for i in range(n):
            jacobian = randn_(batched_shape)
            gramian_accumulator.accumulate_path_jacobians(key, [jacobian])
        assert key not in gramian_accumulator._summed_jacobians.keys()
        assert key not in gramian_accumulator._path_counter.keys()
