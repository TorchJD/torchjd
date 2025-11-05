from math import prod
from typing import cast

import torch
from torch import Tensor, arange, tensor
from torch.ops import aten  # type: ignore

from torchjd.sparse._structured_sparse_tensor import (
    StructuredSparseTensor,
    impl,
    print_fallback,
    to_most_efficient_tensor,
    unwrap_to_dense,
)


@impl(aten.view.default)
def view_default(t: StructuredSparseTensor, shape: list[int]) -> Tensor:
    assert isinstance(t, StructuredSparseTensor)

    shape = infer_shape(shape, t.numel())

    if prod(shape) != t.numel():
        raise ValueError(f"shape '{shape}' is invalid for input of size {t.numel()}")

    new_v_to_ps = []
    idx = 0
    flat_v_to_ps = [dim for dims in t.v_to_ps for dim in dims]
    new_physical = t.physical
    for s in shape:
        group = []
        current_size = 1

        while current_size < s:
            if idx >= len(flat_v_to_ps):
                # TODO: I don't think this can happen, need to review and remove if I'm right.
                raise ValueError()

            pdim = flat_v_to_ps[idx]
            pdim_size = new_physical.shape[pdim]

            if current_size * pdim_size > s:
                # Need to split physical dimension
                if s % current_size != 0:
                    raise ValueError("Can't split physical dimension")

                new_pdim_first_dim_size = s // current_size

                if pdim_size % new_pdim_first_dim_size != 0:
                    raise ValueError("Can't split physical dimension")

                new_pdim_shape = [new_pdim_first_dim_size, pdim_size // new_pdim_first_dim_size]
                new_physical, new_encoding = unsquash_pdim(new_physical, pdim, new_pdim_shape)

                new_v_to_ps = [
                    [new_d for d in dims for new_d in new_encoding[d]] for dims in new_v_to_ps
                ]
                # A bit of a weird trick here. We want to re-encode flat_v_to_ps according to
                # new_encoding. However, re-encoding elements before idx would potentially change
                # the length of the list before idx, so idx would not have the right value anymore.
                # Since we don't need the elements of flat_v_to_ps that are before idx anyway, we
                # just get rid of them and re-encode flat_v_to_ps[idx:] instead, and reset idx to 0
                # to say that we're back at the beginning of this new list.
                flat_v_to_ps = [new_d for d in flat_v_to_ps[idx:] for new_d in new_encoding[d]]
                idx = 0

            group.append(pdim)
            current_size *= new_physical.shape[pdim]
            idx += 1

        new_v_to_ps.append(group)

    if idx != len(flat_v_to_ps):
        raise ValueError(f"idx != len(flat_v_to_ps). {idx}; {flat_v_to_ps}; {shape}; {t.v_to_ps}")

    # The above code does not handle physical dimension squashing, so the physical is not
    # necessarily maximally squashed at this point, so we need the safe constructor.
    return to_most_efficient_tensor(new_physical, new_v_to_ps)


def infer_shape(shape: list[int], numel: int) -> list[int]:
    if shape.count(-1) > 1:
        raise ValueError("Only one dimension can be inferred")
    known = 1
    for s in shape:
        if s != -1:
            known *= s
    inferred = numel // known
    return [inferred if s == -1 else s for s in shape]


def unsquash_pdim_from_strides(
    physical: Tensor, pdim: int, new_pdim_shape: list[int]
) -> tuple[Tensor, Tensor]:
    new_shape = list(physical.shape)
    new_shape = new_shape[:pdim] + new_pdim_shape + new_shape[pdim + 1 :]
    new_physical = physical.reshape(new_shape)

    stride_multipliers = tensor([prod(new_pdim_shape[i + 1 :]) for i in range(len(new_pdim_shape))])
    return new_physical, stride_multipliers


def unsquash_pdim(
    physical: Tensor, pdim: int, new_pdim_shape: list[int]
) -> tuple[Tensor, list[list[int]]]:
    new_shape = list(physical.shape)
    new_shape = new_shape[:pdim] + new_pdim_shape + new_shape[pdim + 1 :]
    new_physical = physical.reshape(new_shape)

    def new_encoding_fn(d: int) -> list[int]:
        if d < pdim:
            return [d]
        elif d > pdim:
            return [d + len(new_pdim_shape) - 1]
        else:
            return [pdim + i for i in range(len(new_pdim_shape))]

    new_encoding = [new_encoding_fn(d) for d in range(len(physical.shape))]
    return new_physical, new_encoding


@impl(aten._unsafe_view.default)
def _unsafe_view_default(t: StructuredSparseTensor, shape: list[int]) -> Tensor:
    return view_default(
        t, shape
    )  # We don't do the optimizations that they do in https://github.com/pytorch/pytorch/blame/main/aten/src/ATen/native/TensorShape.cpp


@impl(aten.unsqueeze.default)
def unsqueeze_default(t: StructuredSparseTensor, dim: int) -> StructuredSparseTensor:
    assert isinstance(t, StructuredSparseTensor)
    assert -t.ndim - 1 <= dim < t.ndim + 1

    if dim < 0:
        dim = t.ndim + dim + 1

    new_strides = torch.concatenate(
        [t.strides[:dim], torch.zeros(1, t.strides.shape[1], dtype=torch.int64), t.strides[dim:]]
    )
    return StructuredSparseTensor(t.physical, new_strides)


@impl(aten.squeeze.dims)
def squeeze_dims(t: StructuredSparseTensor, dims: list[int] | int | None) -> Tensor:
    assert isinstance(t, StructuredSparseTensor)

    if dims is None:
        excluded = set(range(t.ndim))
    elif isinstance(dims, int):
        excluded = {dims}
    else:
        excluded = set(dims)

    is_row_kept = [i not in excluded for i in range(t.ndim)]
    new_strides = t.strides[is_row_kept]
    return to_most_efficient_tensor(t.physical, new_strides)


@impl(aten.permute.default)
def permute_default(t: StructuredSparseTensor, dims: list[int]) -> StructuredSparseTensor:
    new_strides = t.strides[torch.tensor(dims)]
    return StructuredSparseTensor(t.physical, new_strides)


@impl(aten.cat.default)
def cat_default(tensors: list[Tensor], dim: int) -> Tensor:
    if any(not isinstance(t, StructuredSparseTensor) for t in tensors):
        print_fallback(aten.cat.default, (tensors, dim), {})
        return aten.cat.default([unwrap_to_dense(t) for t in tensors])

    tensors_ = [cast(StructuredSparseTensor, t) for t in tensors]
    ref_tensor = tensors_[0]
    ref_strides = ref_tensor.strides
    if any(not torch.equal(t.strides, ref_strides) for t in tensors_[1:]):
        raise NotImplementedError(
            "Override for aten.cat.default does not support SSTs that do not all have the same "
            f"strides. Found the following tensors:\n{[t.debug_info() for t in tensors_]} and the "
            f"following dim: {dim}."
        )

    # We need to try to find the (pretty sure it either does not exist or is unique) physical
    # dimension that makes us only move on virtual dimension dim. It also needs to be such that
    # traversing it entirely brings us exactly to the end of virtual dimension dim.

    ref_virtual_dim_size = ref_tensor.shape[dim]
    indices = torch.argwhere(
        torch.eq(ref_strides[dim] * tensor(ref_tensor.physical.shape), ref_virtual_dim_size)
        & torch.eq(ref_strides.sum(dim=0) * tensor(ref_tensor.physical.shape), ref_virtual_dim_size)
    )
    assert len(indices) <= 1

    if len(indices) == 0:
        # Add a physical dimension pdim on which we can concatenate the physicals such that this
        # translates into a concatenation of the virtuals on virtual dimension dim.

        pdim = ref_tensor.physical.ndim
        physicals = [t.physical.unsqueeze(-1) for t in tensors_]
        new_stride_column = torch.zeros(ref_tensor.ndim, 1, dtype=torch.int64)
        new_stride_column[dim, 0] = ref_virtual_dim_size
        new_strides = torch.concatenate([ref_tensor.strides, new_stride_column], dim=1)
    else:
        # Such a physical dimension already exists. Note that an alternative implementation would be
        # to simply always add the physical dimension, and squash it if it ends up being not needed.
        physicals = [t.physical for t in tensors_]
        pdim = indices[0][0]
        new_strides = ref_tensor.strides

    new_physical = aten.cat.default(physicals, dim=pdim)
    return StructuredSparseTensor(new_physical, new_strides)


@impl(aten.expand.default)
def expand_default(t: StructuredSparseTensor, sizes: list[int]) -> StructuredSparseTensor:
    # note that sizes could also be just an int, or a torch.Size i think
    assert isinstance(t, StructuredSparseTensor)
    assert isinstance(sizes, list)
    assert len(sizes) >= t.ndim

    # Add as many dimensions as needed at the beginning of the tensor (as torch.expand works)
    for _ in range(len(sizes) - t.ndim):
        t = t.unsqueeze(0)

    # Try to expand each dimension to its new size
    new_physical = t.physical
    new_strides = t.strides
    for d, (vstride, orig_size, new_size) in enumerate(zip(t.strides, t.shape, sizes, strict=True)):
        if vstride.sum() > 0 and orig_size != new_size and new_size != -1:
            raise ValueError(
                f"Cannot expand dim {d} of size != 1. Found size {orig_size} and target size "
                f"{new_size}."
            )

        if vstride.sum() == 0 and new_size != 1 and new_size != -1:
            # Add a dimension of size new_size at the end of the physical tensor.
            new_physical_shape = list(new_physical.shape) + [new_size]
            new_physical = new_physical.unsqueeze(-1).expand(new_physical_shape)

            # Make this new physical dimension have a stride of 1 at virtual dimension d and 0 at
            # every other virtual dimension
            new_stride_column = torch.zeros(t.ndim, 1, dtype=torch.int64)
            new_stride_column[d, 0] = 1
            new_strides = torch.cat([new_strides, new_stride_column], dim=1)

    return StructuredSparseTensor(new_physical, new_strides)


@impl(aten.broadcast_tensors.default)
def broadcast_tensors_default(tensors: list[Tensor]) -> tuple[Tensor, Tensor]:
    if len(tensors) != 2:
        raise NotImplementedError()

    t1, t2 = tensors

    if t1.shape == t2.shape:
        return t1, t2

    a = t1 if t1.ndim >= t2.ndim else t2
    b = t2 if t1.ndim >= t2.ndim else t1

    a_shape = list(a.shape)
    padded_b_shape = [1] * (a.ndim - b.ndim) + list(b.shape)

    new_shape = list[int]()

    for s_a, s_b in zip(a_shape, padded_b_shape):
        if s_a != 1 and s_b != 1 and s_a != s_b:
            raise ValueError("Incompatible shapes for broadcasting")
        else:
            new_shape.append(max(s_a, s_b))

    return aten.expand.default(t1, new_shape), aten.expand.default(t2, new_shape)


@impl(aten.transpose.int)
def transpose_int(t: StructuredSparseTensor, dim0: int, dim1: int) -> StructuredSparseTensor:
    assert isinstance(t, StructuredSparseTensor)
    return StructuredSparseTensor(t.physical, _swap_rows(t.strides, dim0, dim1))


def _swap_rows(matrix: Tensor, c0: int, c1: int) -> Tensor:
    index = arange(matrix.shape[0])
    index[c0] = c1
    index[c1] = c0
    return matrix[index]
