import torch
from pytest import mark
from torch.ops import aten  # type: ignore
from torch.testing import assert_close
from utils.tensors import randn_, zeros_

from torchjd.sparse._diagonal_sparse_tensor import (
    _IN_PLACE_POINTWISE_FUNCTIONS,
    _POINTWISE_FUNCTIONS,
    DiagonalSparseTensor,
    einsum,
    encode_by_order,
    fix_ungrouped_dims,
    get_groupings,
    unsquash_pdim,
)


def test_to_dense():
    n = 2
    m = 3
    a = randn_([n, m])
    b = DiagonalSparseTensor(a, [[0], [1], [1], [0]])
    c = b.to_dense()

    for i in range(n):
        for j in range(m):
            assert c[i, j, j, i] == a[i, j]


def test_einsum():
    a = DiagonalSparseTensor(torch.randn([4, 5]), [[0], [0], [1]])
    b = DiagonalSparseTensor(torch.randn([4, 5]), [[0], [1], [1]])

    res = einsum((a, [0, 1, 2]), (b, [0, 2, 3]), output=[0, 1, 3])

    expected = torch.einsum("ijk,ikl->ijl", a.to_dense(), b.to_dense())
    assert_close(res.to_dense(), expected)


@mark.parametrize(
    "shape",
    [
        [],
        [2],
        [2, 3],
        [2, 3, 4],
    ],
)
def test_diagonal_sparse_tensor_scalar(shape: list[int]):
    a = randn_(shape)
    b = DiagonalSparseTensor(a, [[dim] for dim in range(len(shape))])

    assert_close(a, b.to_dense())


@mark.parametrize("dim", [2, 3, 4, 5, 10])
def test_diag_equivalence(dim: int):
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [[0], [0]])

    diag_a = torch.diag(a)

    assert_close(b.to_dense(), diag_a)


def test_three_virtual_single_physical():
    dim = 10
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [[0], [0], [0]])

    expected = zeros_([dim, dim, dim])
    for i in range(dim):
        expected[i, i, i] = a[i]

    assert_close(b.to_dense(), expected)


@mark.parametrize("func", _POINTWISE_FUNCTIONS)
def test_pointwise(func):
    dim = 10
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [[0], [0]])
    c = b.to_dense()
    res = func(b)
    assert isinstance(res, DiagonalSparseTensor)

    assert_close(res.to_dense(), func(c), equal_nan=True)


@mark.parametrize("func", _IN_PLACE_POINTWISE_FUNCTIONS)
def test_inplace_pointwise(func):
    dim = 10
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [[0], [0]])
    c = b.to_dense()
    func(b)
    assert isinstance(b, DiagonalSparseTensor)

    assert_close(b.to_dense(), func(c), equal_nan=True)


@mark.parametrize("func", [torch.mean, torch.sum])
def test_unary(func):
    dim = 10
    a = randn_([dim])
    b = DiagonalSparseTensor(a, [[0], [0]])
    c = b.to_dense()

    res = func(b)
    assert_close(res.to_dense(), func(c))


@mark.parametrize(
    ["physical_shape", "v_to_ps", "target_shape", "expected_physical_shape", "expected_v_to_ps"],
    [
        ([2, 3], [[0], [0], [1]], [2, 2, 3], [2, 3], [[0], [0], [1]]),  # no change of shape
        ([2, 3], [[0], [0, 1]], [2, 6], [2, 3], [[0], [0, 1]]),  # no change of shape
        ([2, 3], [[0], [0], [1]], [2, 6], [2, 3], [[0], [0, 1]]),  # squashing 2 dims
        ([2, 3], [[0], [0, 1]], [2, 2, 3], [2, 3], [[0], [0], [1]]),  # unsquashing into 2 dims
        ([2, 3], [[0, 0, 1]], [2, 6], [2, 3], [[0], [0, 1]]),  # unsquashing into 2 dims
        ([2, 3], [[0], [0], [1]], [12], [2, 3], [[0, 0, 1]]),  # squashing 3 dims
        ([2, 3], [[0, 0, 1]], [2, 2, 3], [2, 3], [[0], [0], [1]]),  # unsquashing into 3 dims
        ([4], [[0], [0]], [2, 2, 4], [2, 2], [[0], [1], [0, 1]]),  # unsquashing physical dim
        ([4], [[0], [0]], [4, 2, 2], [2, 2], [[0, 1], [0], [1]]),  # unsquashing physical dim
        ([2, 3, 4], [[0], [0], [1], [2]], [4, 12], [2, 12], [[0, 0], [1]]),  # world boss
        ([2, 12], [[0, 0], [1]], [2, 2, 3, 4], [2, 3, 4], [[0], [0], [1], [2]]),  # world boss
    ],
)
def test_view(
    physical_shape: list[int],
    v_to_ps: list[list[int]],
    target_shape: list[int],
    expected_physical_shape: list[int],
    expected_v_to_ps: list[list[int]],
):
    a = randn_(tuple(physical_shape))
    t = DiagonalSparseTensor(a, v_to_ps)

    result = aten.view.default(t, target_shape)
    expected = t.to_dense().reshape(target_shape)

    assert isinstance(result, DiagonalSparseTensor)
    assert list(result.physical.shape) == expected_physical_shape
    assert result.v_to_ps == expected_v_to_ps
    assert torch.all(torch.eq(result.to_dense(), expected))


@mark.parametrize(
    ["input", "expected_output", "expected_destination"],
    [
        ([0, 1, 0, 2, 1, 3], [0, 1, 0, 2, 1, 3], [0, 1, 2, 3]),  # trivial
        ([1, 0, 3, 2, 1], [0, 1, 2, 3, 0], [1, 0, 3, 2]),
        ([1, 0, 3, 2], [0, 1, 2, 3], [1, 0, 3, 2]),
        ([0, 2, 0, 1], [0, 1, 0, 2], [0, 2, 1]),
        ([1, 0, 0, 1], [0, 1, 1, 0], [1, 0]),
    ],
)
def test_encode_by_order(
    input: list[int],
    expected_output: list[int],
    expected_destination: list[int],
):
    output, destination = encode_by_order(input)

    assert output == expected_output
    assert destination == expected_destination


@mark.parametrize(
    ["v_to_ps", "expected_groupings"],
    [
        ([[0, 1, 2], [2, 0, 1], [2]], [[0, 1], [2]]),
    ],
)
def test_get_groupings(v_to_ps: list[list[int]], expected_groupings: list[list[int]]):
    groupings = get_groupings(v_to_ps)
    print(groupings)

    assert groupings == expected_groupings


@mark.parametrize(
    ["physical_shape", "v_to_ps", "expected_physical_shape", "expected_v_to_ps"],
    [
        ([3, 4, 5], [[0, 1, 2], [2, 0, 1], [2]], [12, 5], [[0, 1], [1, 0], [1]]),
        ([32, 20, 8], [[0], [1, 0], [2]], [32, 20, 8], [[0], [1, 0], [2]]),
    ],
)
def test_fix_ungrouped_dims(
    physical_shape: list[int],
    v_to_ps: list[list[int]],
    expected_physical_shape: list[int],
    expected_v_to_ps: list[list[int]],
):
    physical = torch.randn(physical_shape)
    fixed_physical, fixed_v_to_ps = fix_ungrouped_dims(physical, v_to_ps)

    assert list(fixed_physical.shape) == expected_physical_shape
    assert fixed_v_to_ps == expected_v_to_ps


@mark.parametrize(
    [
        "physical_shape",
        "pdim",
        "new_pdim_shape",
        "expected_physical_shape",
        "expected_new_encoding",
    ],
    [
        ([4], 0, [4], [4], [[0]]),  # trivial
        ([4], 0, [2, 2], [2, 2], [[0, 1]]),
        ([3, 4, 5], 1, [2, 1, 1, 2], [3, 2, 1, 1, 2, 5], [[0], [1, 2, 3, 4], [5]]),
    ],
)
def test_unsquash_pdim(
    physical_shape: list[int],
    pdim: int,
    new_pdim_shape: list[int],
    expected_physical_shape: list[int],
    expected_new_encoding: list[list[int]],
):
    physical = torch.randn(physical_shape)
    new_physical, new_encoding = unsquash_pdim(physical, pdim, new_pdim_shape)

    assert list(new_physical.shape) == expected_physical_shape
    assert new_encoding == expected_new_encoding
