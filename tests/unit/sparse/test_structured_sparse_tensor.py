import torch
from pytest import mark
from torch.ops import aten  # type: ignore
from torch.testing import assert_close
from utils.tensors import randn_, tensor_, zeros_

from torchjd.sparse._aten_function_overrides.einsum import einsum
from torchjd.sparse._aten_function_overrides.pointwise import (
    _IN_PLACE_POINTWISE_FUNCTIONS,
    _POINTWISE_FUNCTIONS,
)
from torchjd.sparse._aten_function_overrides.shape import unsquash_pdim
from torchjd.sparse._structured_sparse_tensor import (
    StructuredSparseTensor,
    clear_null_stride_columns,
    encode_by_order,
    fix_ungrouped_dims,
    get_groupings,
)


def test_to_dense():
    n = 2
    m = 3
    a = randn_([n, m])
    b = StructuredSparseTensor(a, [[0], [1], [1], [0]])
    c = b.to_dense()

    for i in range(n):
        for j in range(m):
            assert c[i, j, j, i] == a[i, j]


def test_to_dense2():
    a = tensor_([1.0, 2.0, 3.0])
    b = StructuredSparseTensor(a, [[0, 0]])
    c = b.to_dense()
    expected = tensor_([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
    assert torch.all(torch.eq(c, expected))


@mark.parametrize(
    ["a_pshape", "a_v_to_ps", "b_pshape", "b_v_to_ps", "a_indices", "b_indices", "output_indices"],
    [
        ([4, 5], [[0], [0], [1]], [4, 5], [[0], [1], [1]], [0, 1, 2], [0, 2, 3], [0, 1, 3]),
        ([2, 3, 5], [[0, 1], [2, 0]], [10, 3], [[0], [1]], [0, 1], [1, 2], [0, 2]),
        ([2, 3], [[0, 1]], [6], [[0]], [0], [0], []),
        ([6, 2, 3], [[0], [1], [2]], [2, 3], [[0, 1], [0], [1]], [0, 1, 2], [0, 1, 2], [0, 1, 2]),
    ],
)
def test_einsum(
    a_pshape: list[int],
    a_v_to_ps: list[list[int]],
    b_pshape: list[int],
    b_v_to_ps: list[list[int]],
    a_indices: list[int],
    b_indices: list[int],
    output_indices: list[int],
):
    a = StructuredSparseTensor(randn_(a_pshape), a_v_to_ps)
    b = StructuredSparseTensor(randn_(b_pshape), b_v_to_ps)

    res = einsum((a, a_indices), (b, b_indices), output=output_indices)

    expected = torch.einsum(a.to_dense(), a_indices, b.to_dense(), b_indices, output_indices)

    assert isinstance(res, StructuredSparseTensor)
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
def test_structured_sparse_tensor_scalar(shape: list[int]):
    a = randn_(shape)
    b = StructuredSparseTensor(a, [[dim] for dim in range(len(shape))])

    assert_close(a, b.to_dense())


@mark.parametrize("dim", [2, 3, 4, 5, 10])
def test_diag_equivalence(dim: int):
    a = randn_([dim])
    b = StructuredSparseTensor(a, [[0], [0]])

    diag_a = torch.diag(a)

    assert_close(b.to_dense(), diag_a)


def test_three_virtual_single_physical():
    dim = 10
    a = randn_([dim])
    b = StructuredSparseTensor(a, [[0], [0], [0]])

    expected = zeros_([dim, dim, dim])
    for i in range(dim):
        expected[i, i, i] = a[i]

    assert_close(b.to_dense(), expected)


@mark.parametrize("func", _POINTWISE_FUNCTIONS)
def test_pointwise(func):
    dim = 10
    a = randn_([dim])
    b = StructuredSparseTensor(a, [[0], [0]])
    c = b.to_dense()
    res = func(b)
    assert isinstance(res, StructuredSparseTensor)

    assert_close(res.to_dense(), func(c), equal_nan=True)


@mark.parametrize("func", _IN_PLACE_POINTWISE_FUNCTIONS)
def test_inplace_pointwise(func):
    dim = 10
    a = randn_([dim])
    b = StructuredSparseTensor(a, [[0], [0]])
    c = b.to_dense()
    func(b)
    assert isinstance(b, StructuredSparseTensor)

    assert_close(b.to_dense(), func(c), equal_nan=True)


@mark.parametrize("func", [torch.mean, torch.sum])
def test_unary(func):
    dim = 10
    a = randn_([dim])
    b = StructuredSparseTensor(a, [[0], [0]])
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
    t = StructuredSparseTensor(a, v_to_ps)

    result = aten.view.default(t, target_shape)
    expected = t.to_dense().reshape(target_shape)

    assert isinstance(result, StructuredSparseTensor)
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
    ["pshape", "strides", "expected"],
    [
        (
            [[32, 2, 3, 4, 5]],
            torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 60, 20, 5, 1]]),
            [[0], [1, 2, 3, 4]],
        )
    ],
)
def test_get_groupings(pshape: list[int], strides: torch.Tensor, expected: list[list[int]]):
    result = get_groupings(pshape, strides)
    assert result == expected


@mark.parametrize(
    ["physical_shape", "v_to_ps", "expected_physical_shape", "expected_v_to_ps"],
    [
        ([3, 4, 5], [[0, 1, 2], [2, 0, 1], [2]], [12, 5], [[0, 1], [1, 0], [1]]),
        ([32, 20, 8], [[0], [1, 0], [2]], [32, 20, 8], [[0], [1, 0], [2]]),
        ([3, 3, 4], [[0, 1], [1, 2]], [3, 3, 4], [[0, 1], [1, 2]]),
    ],
)
def test_fix_ungrouped_dims(
    physical_shape: list[int],
    v_to_ps: list[list[int]],
    expected_physical_shape: list[int],
    expected_v_to_ps: list[list[int]],
):
    physical = randn_(physical_shape)
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
    physical = randn_(physical_shape)
    new_physical, new_encoding = unsquash_pdim(physical, pdim, new_pdim_shape)

    assert list(new_physical.shape) == expected_physical_shape
    assert new_encoding == expected_new_encoding


@mark.parametrize(
    ["sst_args", "dim"],
    [
        ([([3], [[0], [0]]), ([3], [[0], [0]])], 1),
        ([([3, 2], [[0], [1, 0]]), ([3, 2], [[0], [1, 0]])], 1),
    ],
)
def test_concatenate(
    sst_args: list[tuple[list[int], list[list[int]]]],
    dim: int,
):
    tensors = [StructuredSparseTensor(randn_(pshape), v_to_ps) for pshape, v_to_ps in sst_args]
    res = aten.cat.default(tensors, dim)
    expected = aten.cat.default([t.to_dense() for t in tensors], dim)

    assert isinstance(res, StructuredSparseTensor)
    assert torch.all(torch.eq(res.to_dense(), expected))


@mark.parametrize(
    ["physical", "strides", "expected_physical", "expected_strides"],
    [
        ([[1, 2, 3], [4, 5, 6]], [[1, 0], [1, 0], [2, 0]], [6, 15], [[1], [1], [2]]),
        (
            [[1, 2, 3], [4, 5, 6]],
            [[1, 1], [1, 0], [2, 0]],
            [[1, 2, 3], [4, 5, 6]],
            [[1, 1], [1, 0], [2, 0]],
        ),
    ],
)
def test_clear_null_stride_columns(
    physical: list,
    strides: list,
    expected_physical: list,
    expected_strides: list,
):
    physical, strides = torch.tensor(physical), torch.tensor(strides)
    expected_physical, expected_strides = torch.tensor(expected_physical), torch.tensor(
        expected_strides
    )

    physical, strides = clear_null_stride_columns(physical, strides)
    assert_close(physical, expected_physical)
    assert_close(strides, expected_strides)
