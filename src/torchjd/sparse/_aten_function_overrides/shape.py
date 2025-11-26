import operator
from itertools import accumulate
from math import prod
from typing import cast

import torch
from torch import Tensor, arange, cat, tensor
from torch.ops import aten  # type: ignore

from torchjd.sparse._sparse_latticed_tensor import (
    SparseLatticedTensor,
    impl,
    print_fallback,
    to_most_efficient_tensor,
    unwrap_to_dense,
)


@impl(aten.view.default)
def view_default(t: SparseLatticedTensor, shape: list[int]) -> Tensor:
    """
    The main condition that we want to respect is that the indexing in the flattened virtual
    tensor should remain the same before and after the reshape, i.e.

                                    c.T S = c'.T S'    (1)
    where:
    * c is the reversed vector of cumulative physical shape before the reshape, i.e.
        c.T = [prod(t.shape[1:]), prod(t.shape[2:]), ..., t.shape[-1], 1]
    * c' is the same thing but after the reshape, i.e.
        c'.T = [prod(shape[1:]), prod(shape[2:]), ..., shape[-1], 1]
    * S is the original basis matrix (t.basis)
    * S' is the basis matrix after reshaping.

    For u, v in Z^m and c in Z, say that u ≡ v (mod c) if u_i ≡ v_i (mod c) for all i.
    Note that c'.T S' ≡ S'[-1] (mod shape[-1])
    So if we set S'[-1] = c.T S % shape[-1], we have c.T S ≡ c'.T S' (mod shape[-1])

    (c'.T S' - S'[-1]) // shape[-1] ≡ S'[-1] (mod shape[-1])
    ...
    """

    assert isinstance(t, SparseLatticedTensor)

    shape = infer_shape(shape, t.numel())

    if prod(shape) != t.numel():
        raise ValueError(f"shape '{shape}' is invalid for input of size {t.numel()}")

    S = t.basis
    vshape = list(t.shape)
    c = _reverse_cumulative_product(vshape)
    c_prime = _reverse_cumulative_product(shape)
    new_basis = ((c @ S).unsqueeze(0) // c_prime.unsqueeze(1)) % tensor(shape).unsqueeze(1)
    return to_most_efficient_tensor(t.physical, new_basis)


def _reverse_cumulative_product(values: list[int]) -> Tensor:
    return tensor(list(accumulate((values[1:] + [1])[::-1], operator.mul))[::-1])


def infer_shape(shape: list[int], numel: int) -> list[int]:
    if shape.count(-1) > 1:
        raise ValueError("Only one dimension can be inferred")
    known = 1
    for s in shape:
        if s != -1:
            known *= s
    inferred = numel // known
    return [inferred if s == -1 else s for s in shape]


@impl(aten._unsafe_view.default)
def _unsafe_view_default(t: SparseLatticedTensor, shape: list[int]) -> Tensor:
    return view_default(
        t, shape
    )  # We don't do the optimizations that they do in https://github.com/pytorch/pytorch/blame/main/aten/src/ATen/native/TensorShape.cpp


@impl(aten.unsqueeze.default)
def unsqueeze_default(t: SparseLatticedTensor, dim: int) -> SparseLatticedTensor:
    assert isinstance(t, SparseLatticedTensor)
    assert -t.ndim - 1 <= dim < t.ndim + 1

    if dim < 0:
        dim = t.ndim + dim + 1

    pdims = t.basis.shape[1]
    new_basis = cat([t.basis[:dim], torch.zeros(1, pdims, dtype=torch.int64), t.basis[dim:]])
    new_offset = cat([t.offset[:dim], torch.zeros(1, dtype=torch.int64), t.offset[dim:]])
    new_size = cat([t.size[:dim], torch.zeros(1, dtype=torch.int64), t.size[dim:]])
    return SparseLatticedTensor(t.physical, new_basis, new_offset, new_size)


@impl(aten.squeeze.dims)
def squeeze_dims(t: SparseLatticedTensor, dims: list[int] | int | None) -> Tensor:
    assert isinstance(t, SparseLatticedTensor)
    # TODO: verify that the specified dimensions are of size 1.

    if dims is None:
        excluded = set(range(t.ndim))
    elif isinstance(dims, int):
        excluded = {dims}
    else:
        excluded = set(dims)

    is_row_kept = [i not in excluded for i in range(t.ndim)]
    new_basis = t.basis[is_row_kept]
    new_offset = t.offset[is_row_kept]
    new_size = t.size[is_row_kept]
    return to_most_efficient_tensor(t.physical, new_basis, new_offset, new_size)


@impl(aten.permute.default)
def permute_default(t: SparseLatticedTensor, dims: list[int]) -> SparseLatticedTensor:
    new_basis = t.basis[dims]
    new_offset = t.offset[dims]
    new_size = t.size[dims]
    return SparseLatticedTensor(t.physical, new_basis, new_offset, new_size)


@impl(aten.cat.default)
def cat_default(tensors: list[Tensor], dim: int = 0) -> Tensor:
    if any(not isinstance(t, SparseLatticedTensor) for t in tensors):
        print_fallback(aten.cat.default, (tensors, dim), {})
        return aten.cat.default([unwrap_to_dense(t) for t in tensors])

    tensors_ = [cast(SparseLatticedTensor, t) for t in tensors]
    ref_tensor = tensors_[0]
    ref_basis = ref_tensor.basis
    if any(not torch.equal(t.basis, ref_basis) for t in tensors_[1:]):
        raise NotImplementedError(
            "Override for aten.cat.default does not support SLTs that do not all have the same "
            f"basis. Found the following tensors:\n{[repr(t) for t in tensors_]} and the following "
            f"dim: {dim}."
        )
    if any(t.physical.shape != ref_tensor.physical.shape for t in tensors_[1:]):
        # This can happen in the following example:
        # t1 = SLT([1 2 3], [[2]])
        # t2 = SLT([4 5 6 7], [[2]])
        # The expected result would be 1 0 2 0 3 4 0 5 0 6 0 7, but this is not representable
        # efficiently as an SLT (because there is no 0 between 3 and 4, and both physicals have a
        # different shape so we can't just stack them).

        # TODO: Maybe a partial densify is possible rather than a full densify.
        print_fallback(aten.cat.default, (tensors, dim), {})
        return aten.cat.default([unwrap_to_dense(t) for t in tensors])

    # We need to try to find the (pretty sure it either does not exist or is unique) physical
    # dimension that makes us only move on virtual dimension dim. It also needs to be such that
    # traversing it entirely brings us exactly to the end of virtual dimension dim.

    ref_virtual_dim_size = ref_tensor.shape[dim]
    indices = torch.argwhere(
        torch.eq(ref_basis[dim] * tensor(ref_tensor.physical.shape), ref_virtual_dim_size)
        & torch.eq(ref_basis.sum(dim=0) * tensor(ref_tensor.physical.shape), ref_virtual_dim_size)
    )
    assert len(indices) <= 1

    if len(indices) == 0:
        # Add a physical dimension pdim on which we can concatenate the physicals such that this
        # translates into a concatenation of the virtuals on virtual dimension dim.

        pdim = ref_tensor.physical.ndim
        physicals = [t.physical.unsqueeze(-1) for t in tensors_]
        new_basis_vector = torch.zeros(ref_tensor.ndim, 1, dtype=torch.int64)
        new_basis_vector[dim, 0] = ref_virtual_dim_size
        new_basis = torch.concatenate([ref_tensor.basis, new_basis_vector], dim=1)
    else:
        # Such a physical dimension already exists. Note that an alternative implementation would be
        # to simply always add the physical dimension, and squash it if it ends up being not needed.
        physicals = [t.physical for t in tensors_]
        pdim = cast(int, indices[0, 0].item())
        new_basis = ref_tensor.basis

    new_physical = aten.cat.default(physicals, dim=pdim)
    return SparseLatticedTensor(new_physical, new_basis)


@impl(aten.expand.default)
def expand_default(t: SparseLatticedTensor, sizes: list[int]) -> SparseLatticedTensor:
    # note that sizes could also be just an int, or a torch.Size i think
    assert isinstance(t, SparseLatticedTensor)
    assert isinstance(sizes, list)
    assert len(sizes) >= t.ndim

    # Add as many dimensions as needed at the beginning of the tensor (as torch.expand works)
    for _ in range(len(sizes) - t.ndim):
        t = unsqueeze_default(t, 0)

    # Try to expand each dimension to its new size
    new_physical = t.physical
    new_basis = t.basis
    new_sizes = t.size
    for d, (v, orig_size, new_size) in enumerate(zip(t.basis, t.shape, sizes, strict=True)):
        if v.sum() > 0 and orig_size != new_size and new_size != -1:
            raise ValueError(
                f"Cannot expand dim {d} of size != 1. Found size {orig_size} and target size "
                f"{new_size}."
            )

        if v.sum() == 0 and new_size != 1 and new_size != -1:
            # Add a dimension of size new_size at the end of the physical tensor.
            new_physical_shape = list(new_physical.shape) + [new_size]
            new_physical = new_physical.unsqueeze(-1).expand(new_physical_shape)

            # Make the basis vector of this new physical dimension be 1 at virtual dimension d and 0
            # at every other virtual dimension
            new_basis_vector = torch.zeros(t.ndim, 1, dtype=torch.int64)
            new_basis_vector[d, 0] = 1
            new_basis = torch.cat([new_basis, new_basis_vector], dim=1)
            new_sizes[d] = new_size

    return SparseLatticedTensor(new_physical, new_basis, t.offset, new_sizes)


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
def transpose_int(t: SparseLatticedTensor, dim0: int, dim1: int) -> SparseLatticedTensor:
    assert isinstance(t, SparseLatticedTensor)

    new_index = arange(t.basis.shape[0])
    new_index[dim0] = dim1
    new_index[dim1] = dim0

    new_basis = t.basis[new_index]
    new_offset = t.offset[new_index]
    new_shape = list(tensor(t.shape, dtype=torch.int64)[new_index])

    return SparseLatticedTensor(t.physical, new_basis, new_offset, new_shape)
