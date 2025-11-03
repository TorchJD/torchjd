import torch
from torch import Tensor, tensor
from torch.ops import aten  # type: ignore

from torchjd.sparse._structured_sparse_tensor import (
    StructuredSparseTensor,
    impl,
    p_to_vs_from_v_to_ps,
    to_most_efficient_tensor,
    to_structured_sparse_tensor,
)


def prepare_for_elementwise_op(
    t1: Tensor | int | float, t2: Tensor | int | float
) -> tuple[StructuredSparseTensor, StructuredSparseTensor]:
    """
    Prepares two SSTs of the same shape from two args, one of those being a SST, and the other being
    a SST, Tensor, int or float.
    """

    assert isinstance(t1, StructuredSparseTensor) or isinstance(t2, StructuredSparseTensor)

    if isinstance(t1, int) or isinstance(t1, float):
        t1_ = tensor(t1, device=t2.device)
    else:
        t1_ = t1

    if isinstance(t2, int) or isinstance(t2, float):
        t2_ = tensor(t2, device=t1.device)
    else:
        t2_ = t2

    t1_, t2_ = aten.broadcast_tensors.default([t1_, t2_])
    t1_ = to_structured_sparse_tensor(t1_)
    t2_ = to_structured_sparse_tensor(t2_)

    return t1_, t2_


@impl(aten.mul.Tensor)
def mul_Tensor(t1: Tensor | int | float, t2: Tensor | int | float) -> Tensor:
    # Element-wise multiplication with broadcasting
    t1_, t2_ = prepare_for_elementwise_op(t1, t2)
    all_dims = list(range(t1_.ndim))
    return einsum((t1_, all_dims), (t2_, all_dims), output=all_dims)


@impl(aten.div.Tensor)
def div_Tensor(t1: Tensor | int | float, t2: Tensor | int | float) -> Tensor:
    t1_, t2_ = prepare_for_elementwise_op(t1, t2)
    t2_ = StructuredSparseTensor(1.0 / t2_.physical, t2_.v_to_ps)
    all_dims = list(range(t1_.ndim))
    return einsum((t1_, all_dims), (t2_, all_dims), output=all_dims)


@impl(aten.mul.Scalar)
def mul_Scalar(t: StructuredSparseTensor, scalar) -> StructuredSparseTensor:
    # TODO: maybe it could be that scalar is a scalar SST and t is a normal tensor. Need to check
    #  that

    assert isinstance(t, StructuredSparseTensor)
    new_physical = aten.mul.Scalar(t.physical, scalar)
    return StructuredSparseTensor(new_physical, t.v_to_ps)


@impl(aten.add.Tensor)
def add_Tensor(
    t1: Tensor | int | float, t2: Tensor | int | float, alpha: Tensor | float = 1.0
) -> StructuredSparseTensor:
    t1_, t2_ = prepare_for_elementwise_op(t1, t2)

    if t1_.v_to_ps == t2_.v_to_ps:
        new_physical = t1_.physical + t2_.physical * alpha
        return StructuredSparseTensor(new_physical, t1_.v_to_ps)
    else:
        raise NotImplementedError()


def einsum(*args: tuple[StructuredSparseTensor, list[int]], output: list[int]) -> Tensor:

    # First part of the algorithm, determine how to cluster physical indices as well as the common
    # p_shapes corresponding to matching v_dims. Second part translates to physical einsum.

    # get a map from einsum index to (tensor_idx, v_dims)
    # get a map from einsum index to merge of strides corresponding to v_dims with that index
    # use to_target_physical_strides on each physical and v_to_ps
    # cluster pairs of (einsum_index, new_stride) using new_v_to_ps and possibly its corresponding
    #  p_to_vs
    # get unique indices
    # map output indices (there can be splits)
    # call physical einsum
    # build resulting sst

    # OVER

    # an index in the physical einsum is uniquely characterized by a virtual einsum index and a
    # stride corresponding to the physical stride in the virtual one (note that as the virtual shape
    # for two virtual index that match should match, then we want to match the strides and reshape
    # accordingly).
    # We want to cluster such indices whenever several appear in the same p_to_vs

    # TODO: Handle ellipsis
    # If we have an index v for some virtual dim whose corresponding v_to_ps is a non-trivial list
    # [p_1, ..., p_k], then we have to create fresh sub-indices for each dimension.
    # For this reason, an index is decomposed into sub-indices that are then independently
    # clustered.
    # So if an index i in args for some StructuredSparseTensor corresponds to a v_to_ps [j, k, l],
    # We will consider three indices (i, 0), (i, 1) and (i, 2).
    # If furthermore [k] correspond to the v_to_ps of some other tensor with index j, then
    # (i, 1) and (j, 0) will be clustered together (and end up being mapped to the same indice in
    # the resulting einsum).
    # Note that this is a problem if two virtual dimensions (from possibly different
    # StructuredSparseTensors) have the same size but not the same decomposition into physical
    # dimension sizes. For now lets leave the responsibility to care about that in the calling
    # functions, if we can factor code later on we will.

    index_parents = dict[tuple[int, int], tuple[int, int]]()

    def get_representative(index: tuple[int, int]) -> tuple[int, int]:
        if index not in index_parents:
            # If an index is not yet in a cluster, put it in its own.
            index_parents[index] = index
        current = index_parents[index]
        if current != index:
            # Compress path to representative
            index_parents[index] = get_representative(current)
        return index_parents[index]

    def group_indices(indices: list[tuple[int, int]]) -> None:
        first_representative = get_representative(indices[0])
        for i in indices[1:]:
            curr_representative = get_representative(i)
            index_parents[curr_representative] = first_representative

    new_indices_pair = list[list[tuple[int, int]]]()
    physicals = list[Tensor]()
    indices_to_n_pdims = dict[int, int]()
    for t, indices in args:
        assert isinstance(t, StructuredSparseTensor)
        physicals.append(t.physical)
        for pdims, index in zip(t.v_to_ps, indices):
            if index in indices_to_n_pdims:
                assert indices_to_n_pdims[index] == len(pdims)
            else:
                indices_to_n_pdims[index] = len(pdims)
        p_to_vs = p_to_vs_from_v_to_ps(t.v_to_ps)
        for indices_ in p_to_vs:
            # elements in indices[indices_] map to the same dimension, they should be clustered
            # together
            group_indices([(indices[i], sub_i) for i, sub_i in indices_])
        # record the physical dimensions, index[v] for v in vs will end-up mapping to the same
        # final dimension as they were just clustered, so we can take the first, which exists as
        # t is a valid SST.
        new_indices_pair.append([(indices[vs[0][0]], vs[0][1]) for vs in p_to_vs])

    current = 0
    pair_to_int = dict[tuple[int, int], int]()

    def unique_int(pair: tuple[int, int]) -> int:
        nonlocal current
        if pair in pair_to_int:
            return pair_to_int[pair]
        pair_to_int[pair] = current
        current += 1
        return pair_to_int[pair]

    new_indices = [
        [unique_int(get_representative(i)) for i in indices] for indices in new_indices_pair
    ]
    new_output = list[int]()
    v_to_ps = list[list[int]]()
    for i in output:
        current_v_to_ps = []
        for j in range(indices_to_n_pdims[i]):
            k = unique_int(get_representative((i, j)))
            if k in new_output:
                current_v_to_ps.append(new_output.index(k))
            else:
                current_v_to_ps.append(len(new_output))
                new_output.append(k)
        v_to_ps.append(current_v_to_ps)

    physical = torch.einsum(*[x for y in zip(physicals, new_indices) for x in y], new_output)
    # Need to use the safe constructor, otherwise the dimensions may not be maximally grouped.
    # Maybe there is a way to fix that though.
    return to_most_efficient_tensor(physical, v_to_ps)


@impl(aten.bmm.default)
def bmm_default(mat1: Tensor, mat2: Tensor) -> Tensor:
    assert isinstance(mat1, StructuredSparseTensor) or isinstance(mat2, StructuredSparseTensor)
    assert (
        mat1.ndim == 3
        and mat2.ndim == 3
        and mat1.shape[0] == mat2.shape[0]
        and mat1.shape[2] == mat2.shape[1]
    )

    mat1_ = to_structured_sparse_tensor(mat1)
    mat2_ = to_structured_sparse_tensor(mat2)

    # TODO: Verify that the dimension `0` of mat1_ and mat2_ have the same physical dimension sizes
    #  decompositions. If not, can reshape to common decomposition?
    return einsum((mat1_, [0, 1, 2]), (mat2_, [0, 2, 3]), output=[0, 1, 3])


@impl(aten.mm.default)
def mm_default(mat1: Tensor, mat2: Tensor) -> Tensor:
    assert isinstance(mat1, StructuredSparseTensor) or isinstance(mat2, StructuredSparseTensor)
    assert mat1.ndim == 2 and mat2.ndim == 2 and mat1.shape[1] == mat2.shape[0]

    mat1_ = to_structured_sparse_tensor(mat1)
    mat2_ = to_structured_sparse_tensor(mat2)

    return einsum((mat1_, [0, 1]), (mat2_, [1, 2]), output=[0, 2])


@impl(aten.mean.default)
def mean_default(t: StructuredSparseTensor) -> Tensor:
    assert isinstance(t, StructuredSparseTensor)
    return aten.sum.default(t.physical) / t.numel()


@impl(aten.sum.default)
def sum_default(t: StructuredSparseTensor) -> Tensor:
    assert isinstance(t, StructuredSparseTensor)
    return aten.sum.default(t.physical)


@impl(aten.sum.dim_IntList)
def sum_dim_IntList(
    t: StructuredSparseTensor, dim: list[int], keepdim: bool = False, dtype=None
) -> Tensor:
    assert isinstance(t, StructuredSparseTensor)

    if dtype:
        raise NotImplementedError()

    all_dims = list(range(t.ndim))
    result = einsum((t, all_dims), output=[d for d in all_dims if d not in dim])

    if keepdim:
        for d in dim:
            result = result.unsqueeze(d)

    return result
