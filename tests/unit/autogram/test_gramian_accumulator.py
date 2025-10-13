from pytest import mark
from torch import nn
from torch.testing import assert_close
from utils.tensors import randn_, zeros_

from torchjd.autogram._gramian_accumulator import GramianAccumulator


class FakeModule(nn.Module):
    pass


@mark.parametrize(
    ["sizes", "number_of_jacobians"],
    [
        ([4, 7], [3, 7]),
        ([3, 8, 4], [0, 7, 1]),
    ],
)
def test_adding_jacobians_one_by_one(sizes: list[int], number_of_jacobians: list[int]):
    batch_size = 10
    gramian_accumulator = GramianAccumulator()

    keys = [FakeModule() for _ in sizes]
    for key, n in zip(keys, number_of_jacobians):
        for _ in range(n):
            gramian_accumulator.track_module_paths(key)

    expected_gramian = zeros_([batch_size, batch_size])

    for key, size, n in zip(keys, sizes, number_of_jacobians):
        batched_shape = [batch_size, size]
        cumulated_jacobian = zeros_(batched_shape)
        for i in range(n):
            jacobian = randn_(batched_shape)
            gramian_accumulator.accumulate_path_jacobian(key, jacobian)
            cumulated_jacobian += jacobian
        jacobian_matrix = cumulated_jacobian.reshape([batch_size, -1])
        expected_gramian.addmm_(jacobian_matrix, jacobian_matrix.T)

    gramian = gramian_accumulator.gramian
    assert_close(gramian, expected_gramian, rtol=5e-06, atol=2e-05)


def test_returns_none_if_no_jacobian_were_provided():
    gramian_accumulator = GramianAccumulator()
    assert gramian_accumulator.gramian is None


@mark.parametrize(
    ["sizes", "number_of_jacobians"],
    [
        ([5, 7], [3, 7]),
        ([3, 8, 4], [0, 7, 1]),
    ],
)
def test_internal_dicts_are_cleaned(sizes: list[int], number_of_jacobians: list[int]):
    batch_size = 10
    gramian_accumulator = GramianAccumulator()

    keys = [FakeModule() for shape in sizes]
    for key, n in zip(keys, number_of_jacobians):
        for _ in range(n):
            gramian_accumulator.track_module_paths(key)

    for key, size, n in zip(keys, sizes, number_of_jacobians):
        batched_shape = [batch_size, size]
        for i in range(n):
            jacobian = randn_(batched_shape)
            gramian_accumulator.accumulate_path_jacobian(key, jacobian)
        assert key not in gramian_accumulator._summed_jacobians.keys()
        assert key not in gramian_accumulator._path_counter.keys()
