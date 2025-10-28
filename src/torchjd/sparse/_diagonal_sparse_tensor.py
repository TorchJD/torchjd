import operator
from functools import wraps
from itertools import accumulate
from math import prod

import torch
from torch import Tensor
from torch.ops import aten  # type: ignore
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


class DiagonalSparseTensor(torch.Tensor):
    _HANDLED_FUNCTIONS = dict()

    @staticmethod
    def __new__(cls, physical: Tensor, v_to_ps: list[list[int]]):
        # At the moment, this class is not compositional, so we assert
        # that the tensor we're wrapping is exactly a Tensor
        assert type(physical) is Tensor

        # Note [Passing requires_grad=true tensors to subclasses]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calling _make_subclass directly in an autograd context is
        # never the right thing to do, as this will detach you from
        # the autograd graph.  You must create an autograd function
        # representing the "constructor" (NegativeView, in this case)
        # and call that instead.  This assert helps prevent direct usage
        # (which is bad!)
        assert not physical.requires_grad or not torch.is_grad_enabled()

        shape = [prod(physical.shape[i] for i in dims) for dims in v_to_ps]
        return Tensor._make_wrapper_subclass(
            cls, shape, dtype=physical.dtype, device=physical.device
        )

    def __init__(self, physical: Tensor, v_to_ps: list[list[int]]):
        """
        This constructor is made for specifying physical and v_to_ps exactly. It should not modify
        it.

        For this reason, another constructor will be made to either modify the physical / v_to_ps to
        simplify the result, or to create a dense tensor directly if it's already dense. It could
        also be responsible for sorting the first apparition of each physical dim in the flattened
        v_to_ps.
        """

        if any(s == 1 for s in physical.shape):
            raise ValueError(
                "physical must not contain any dimension of size 1. Found physical.shape="
                f"{physical.shape}."
            )
        if not all(all(0 <= dim < physical.ndim for dim in dims) for dims in v_to_ps):
            raise ValueError(
                f"Elements in v_to_ps must map to dimensions in physical. Found {v_to_ps}."
            )
        if len(set().union(*[set(dims) for dims in v_to_ps])) != physical.ndim:
            raise ValueError("Every dimension in physical must appear at least once in v_to_ps.")

        if v_to_ps != encode_v_to_ps(v_to_ps)[0]:
            raise ValueError(
                f"v_to_ps elements are not encoded by first appearance. Found {v_to_ps}."
            )

        if any(len(group) != 1 for group in get_groupings(v_to_ps)):
            raise ValueError(f"Dimensions must be maximally grouped. Found {v_to_ps}.")

        self.physical = physical
        self.v_to_ps = v_to_ps

    def to_dense(
        self, dtype: torch.dtype | None = None, *, masked_grad: bool | None = None
    ) -> Tensor:
        assert dtype is None  # We may add support for this later
        assert masked_grad is None  # We may add support for this later

        if self.physical.ndim == 0:
            return self.physical

        # This is a list of strides whose shape matches that of v_to_ps except that each element
        # is the stride factor of the index to get the right element for the corresponding virtual
        # dimension. Stride is the jump necessary to go from one element to the next one in the
        # specified dimension. For instance if the i'th element of v_to_ps is [0, 1, 2], then the
        # i'th element of _strides is [physical.shape[1] * physical.shape[2], physical.shape[2], 1]
        # and so, if we index dimension i with j=j_0 * stride[0] + j_1 * stride[1] + j_2 * stride[2]
        # which isa unique decomposition, then this corresponds to indexing dimensions v_to_ps[i] at
        # indices [j_0, j_1, j_2]
        s = self.physical.shape
        strides = [
            list(accumulate([1] + [s[dim] for dim in dims[:0:-1]], operator.mul))[::-1]
            for dims in self.v_to_ps
        ]

        p_index_ranges = [torch.arange(s, device=self.physical.device) for s in self.physical.shape]
        p_indices_grid = torch.meshgrid(*p_index_ranges, indexing="ij")
        v_indices_grid = list[Tensor]()
        for stride, dims in zip(strides, self.v_to_ps):
            stride_ = torch.tensor(stride, device=self.physical.device, dtype=torch.int)
            v_indices_grid.append(
                torch.sum(torch.stack([p_indices_grid[d] for d in dims], dim=-1) * stride_, dim=-1)
            )
            # This is supposed to be a vector of shape d_1 * d_2 ...
            # whose elements are the coordinates 1 in p_indices_grad[d_1] times stride 1
            # plus coordinates 2 in p_indices_grad[d_2] times stride 2, etc...

        res = torch.zeros(self.shape, device=self.physical.device, dtype=self.physical.dtype)
        res[tuple(v_indices_grid)] = self.physical
        return res

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in cls._HANDLED_FUNCTIONS:
            return cls._HANDLED_FUNCTIONS[func](*args, **kwargs)

        # --- Fallback: Fold to Dense Tensor ---
        def unwrap_to_dense(t: Tensor):
            if isinstance(t, cls):
                return t.to_dense()
            else:
                return t

        print(f"Falling back to dense for {func.__name__}")
        if len(args) > 0:
            print("* args:")
            for arg in args:
                if isinstance(arg, Tensor):
                    print(f"  > {arg.__class__.__name__} - {arg.shape}")
                else:
                    print(f"  > {arg}")
        if len(kwargs) > 0:
            print("* kwargs:")
            for k, v in kwargs.items():
                print(f"  > {k}: {v}")
        print()
        return func(*tree_map(unwrap_to_dense, args), **tree_map(unwrap_to_dense, kwargs))

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"DiagonalSparseTensor(physical={self.physical}, v_to_ps={self.v_to_ps})"

    def debug_info(self) -> str:
        info = (
            f"shape: {self.shape}\n"
            f"stride(): {self.stride()}\n"
            f"v_to_ps: {self.v_to_ps}\n"
            f"physical.shape: {self.physical.shape}\n"
            f"physical.stride(): {self.physical.stride()}"
        )
        return info

    @classmethod
    def implements(cls, torch_function):
        """Register a torch function override for ScalarTensor"""

        @wraps(torch_function)
        def decorator(func):
            cls._HANDLED_FUNCTIONS[torch_function] = func
            return func

        return decorator


def p_to_vs_from_v_to_ps(v_to_ps: list[list[int]]) -> list[list[tuple[int, int]]]:
    """
    A physical dimension is mapped to a list of couples of the form
    (virtual_dim, sub_index_in_virtual_dim)
    """

    res = dict[int, list[tuple[int, int]]]()
    for v_dim, p_dims in enumerate(v_to_ps):
        for i, p_dim in enumerate(p_dims):
            if p_dim not in res:
                res[p_dim] = [(v_dim, i)]
            else:
                res[p_dim].append((v_dim, i))
    return [res[i] for i in range(len(res))]


def get_groupings(v_to_ps: list[list[int]]) -> list[list[int]]:
    """Example: [[0, 1, 2], [2, 0, 1], [2]] => [[0, 1], [2]]"""

    mapping = dict[int, list[int]]()

    for p_dims in v_to_ps:
        for i, p_dim in enumerate(p_dims):
            if p_dim not in mapping:
                mapping[p_dim] = p_dims[i:]
            else:
                mapping[p_dim] = longest_common_prefix(mapping[p_dim], p_dims[i:])

    groups = []
    visited_is = set()
    for i, group in mapping.items():
        if i in visited_is:
            continue

        groups.append(group)
        visited_is.update(set(group))

    return groups


def longest_common_prefix(l1: list[int], l2: list[int]) -> list[int]:
    prefix = []
    for a, b in zip(l1, l2, strict=False):
        if a == b:
            prefix.append(a)
        else:
            break
    return prefix


def encode_by_order(input: list[int]) -> tuple[list[int], list[int]]:
    """
    Encodes values based on the order of their first appearance, starting at 0 and incrementing.

    Returns the encoded list and the destination mapping each original int to its new encoding.
    destination[i] = j means that all elements of value i in input are mapped to j in the encoded
    list.

    The input list should only contain consecutive integers starting at 0.

    Examples:
        [1, 0, 3, 2] => [0, 1, 2, 3], [1, 0, 3, 2]
        [0, 2, 0, 1] => [0, 1, 0, 2], [0, 2, 1]
        [1, 0, 0, 1] => [0, 1, 1, 0], [1, 0]
    """

    mapping = dict[int, int]()
    curr = 0
    output = []
    for v in input:
        if v not in mapping:
            mapping[v] = curr
            curr += 1
        output.append(mapping[v])
    destination = [mapping[i] for i in range(len(mapping))]

    return output, destination


def encode_v_to_ps(v_to_ps: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    flat_v_to_ps, spec = tree_flatten(v_to_ps)
    sorted_flat_v_to_ps, destination = encode_by_order(flat_v_to_ps)
    return tree_unflatten(sorted_flat_v_to_ps, spec), destination


def to_diagonal_sparse_tensor(t: Tensor) -> DiagonalSparseTensor:
    if isinstance(t, DiagonalSparseTensor):
        return t
    else:
        return make_dst(t, [[i] for i in range(t.ndim)])


def fix_dim_encoding(physical: Tensor, v_to_ps: list[list[int]]) -> tuple[Tensor, list[list[int]]]:
    v_to_ps, destination = encode_v_to_ps(v_to_ps)
    source = list(range(physical.ndim))
    physical = physical.movedim(source, destination)

    return physical, v_to_ps


def fix_dim_of_size_1(physical: Tensor, v_to_ps: list[list[int]]) -> tuple[Tensor, list[list[int]]]:
    is_of_size_1 = [s == 1 for s in physical.shape]

    def new_encoding(d: int) -> int:
        n_removed_dims_before_d = sum(is_of_size_1[:d])
        return d - n_removed_dims_before_d

    physical = physical.squeeze()
    v_to_ps = [[new_encoding(d) for d in dims if not is_of_size_1[d]] for dims in v_to_ps]

    return physical, v_to_ps


def fix_ungrouped_dims(
    physical: Tensor, v_to_ps: list[list[int]]
) -> tuple[Tensor, list[list[int]]]:
    groups = get_groupings(v_to_ps)
    physical = physical.reshape([prod([physical.shape[dim] for dim in group]) for group in groups])
    mapping = {group[0]: i for i, group in enumerate(groups)}
    new_v_to_ps = [[mapping[i] for i in dims if i in mapping] for dims in v_to_ps]

    return physical, new_v_to_ps


def make_dst(physical: Tensor, v_to_ps: list[list[int]]) -> DiagonalSparseTensor:
    """Fix physical and v_to_ps and create a DiagonalSparseTensor with them."""

    physical, v_to_ps = fix_dim_encoding(physical, v_to_ps)
    physical, v_to_ps = fix_dim_of_size_1(physical, v_to_ps)
    physical, v_to_ps = fix_ungrouped_dims(physical, v_to_ps)
    return DiagonalSparseTensor(physical, v_to_ps)


@DiagonalSparseTensor.implements(aten.mean.default)
def mean_default(t: DiagonalSparseTensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t.physical) / t.numel()


@DiagonalSparseTensor.implements(aten.sum.default)
def sum_default(t: DiagonalSparseTensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t.physical)


@DiagonalSparseTensor.implements(aten.pow.Tensor_Scalar)
def pow_Tensor_Scalar(t: DiagonalSparseTensor, exponent: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    if exponent <= 0.0:
        # Need to densify because we don't have pow(0.0, exponent) = 0.0
        return aten.pow.Tensor_Scalar(t.to_dense(), exponent)

    new_physical = aten.pow.Tensor_Scalar(t.physical, exponent)
    return DiagonalSparseTensor(new_physical, t.v_to_ps)


# Somehow there's no pow_.Tensor_Scalar and pow_.Scalar takes tensor and scalar.
@DiagonalSparseTensor.implements(aten.pow_.Scalar)
def pow__Scalar(t: DiagonalSparseTensor, exponent: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    if exponent <= 0.0:
        # Need to densify because we don't have pow(0.0, exponent) = 0.0
        # Note sure if it's even possible to densify in-place, so let's just raise an error.
        raise ValueError(f"in-place pow with an exponent of {exponent} (<= 0) is not supported.")

    aten.pow_.Scalar(t.physical, exponent)
    return t


@DiagonalSparseTensor.implements(aten.unsqueeze.default)
def unsqueeze_default(t: DiagonalSparseTensor, dim: int) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)
    assert -t.ndim - 1 <= dim < t.ndim + 1

    if dim < 0:
        dim = t.ndim + dim + 1

    new_v_to_ps = [p for p in t.v_to_ps]  # Deepcopy the list to not modify the original v_to_ps
    new_v_to_ps.insert(dim, [])

    return DiagonalSparseTensor(t.physical, new_v_to_ps)


@DiagonalSparseTensor.implements(aten.view.default)
def view_default(t: DiagonalSparseTensor, shape: list[int]) -> DiagonalSparseTensor:
    # TODO: add error message when error is raised
    # TODO: handle case where the physical has to be reshaped

    assert isinstance(t, DiagonalSparseTensor)

    if prod(shape) != t.numel():
        raise ValueError(f"shape '{shape}' is invalid for input of size {t.numel()}")

    new_v_to_ps = []
    idx = 0
    flat_v_to_ps = [dim for dims in t.v_to_ps for dim in dims]
    p_shape = t.physical.shape
    for s in shape:
        group = []
        current_product = 1

        while current_product < s:
            if idx >= len(flat_v_to_ps):
                raise ValueError()

            group.append(flat_v_to_ps[idx])
            current_product *= p_shape[flat_v_to_ps[idx]]
            idx += 1

            if current_product > s:
                raise ValueError()

        new_v_to_ps.append(group)

    if idx != len(flat_v_to_ps):
        raise ValueError(f"idx != len(flat_v_to_ps). {idx}; {flat_v_to_ps}; {shape}; {t.v_to_ps}")

    return DiagonalSparseTensor(t.physical, new_v_to_ps)


@DiagonalSparseTensor.implements(aten.expand.default)
def expand_default(t: DiagonalSparseTensor, sizes: list[int]) -> DiagonalSparseTensor:
    # note that sizes could also be just an int, or a torch.Size i think
    assert isinstance(t, DiagonalSparseTensor)
    assert isinstance(sizes, list)
    assert len(sizes) >= t.ndim

    for _ in range(len(sizes) - t.ndim):
        t = t.unsqueeze(0)

    assert len(sizes) == t.ndim

    new_physical = t.physical
    new_v_to_ps = t.v_to_ps
    n_added_physical_dims = 0
    for dim, (ps, orig_size, new_size) in enumerate(zip(t.v_to_ps, t.shape, sizes, strict=True)):
        if len(ps) > 0 and orig_size != new_size and new_size != -1:
            raise ValueError(
                f"Cannot expand dim {dim} of size != 1. Found size {orig_size} and target size "
                f"{new_size}."
            )

        if len(ps) == 0 and new_size != 1 and new_size != -1:
            # Add a dimension of size new_size at the end of the physical tensor.
            new_physical_shape = list(new_physical.shape) + [new_size]
            new_physical = new_physical.unsqueeze(-1).expand(new_physical_shape)
            new_v_to_ps[dim] = [t.physical.ndim + n_added_physical_dims]
            n_added_physical_dims += 1

    new_v_to_ps, destination = encode_v_to_ps(new_v_to_ps)
    new_physical = new_physical.movedim(list(range(len(destination))), destination)

    return DiagonalSparseTensor(new_physical, new_v_to_ps)


@DiagonalSparseTensor.implements(aten.div.Scalar)
def div_Scalar(t: DiagonalSparseTensor, divisor: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    new_physical = aten.div.Scalar(t.physical, divisor)
    return DiagonalSparseTensor(new_physical, t.v_to_ps)


@DiagonalSparseTensor.implements(aten.slice.Tensor)
def slice_Tensor(
    t: DiagonalSparseTensor, dim: int, start: int | None, end: int | None, step: int = 1
) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    physical_dims = t.v_to_ps[dim]

    if len(physical_dims) != 1:
        raise ValueError("Cannot yet slice virtual dim corresponding to several physical dims.")

    physical_dim = physical_dims[0]

    new_physical = aten.slice.Tensor(t.physical, physical_dim, start, end, step)

    return DiagonalSparseTensor(new_physical, t.v_to_ps)


@DiagonalSparseTensor.implements(aten.mul.Tensor)
def mul_Tensor(t1: Tensor, t2: Tensor) -> DiagonalSparseTensor:
    # Element-wise multiplication
    assert isinstance(t1, DiagonalSparseTensor) or isinstance(t2, DiagonalSparseTensor)

    new_physical = aten.mul.Tensor(t1, t2.physical)
    return DiagonalSparseTensor(new_physical, t2.v_to_ps)


@DiagonalSparseTensor.implements(aten.transpose.int)
def transpose_int(t: DiagonalSparseTensor, dim0: int, dim1: int) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    new_v_to_ps = [dims for dims in t.v_to_ps]
    new_v_to_ps[dim0] = t.v_to_ps[dim1]
    new_v_to_ps[dim1] = t.v_to_ps[dim0]

    return DiagonalSparseTensor(t.physical, new_v_to_ps)


def einsum(
    *args: tuple[DiagonalSparseTensor, list[int]], output: list[int]
) -> DiagonalSparseTensor:
    # TODO: Handle ellipsis
    # If we have an index v for some virtual dim whose corresponding v_to_ps is a non-trivial list
    # [p_1, ..., p_k], then we have to create fresh sub-indices for each dimension.
    # For this reason, an index is decomposed into sub-indices that are then independently
    # clustered.
    # So if an index i in args for some DiagonalSparseTensor corresponds to a v_to_ps [j, k, l],
    # We will consider three indices (i, 0), (i, 1) and (i, 2).
    # If furthermore [k] correspond to the v_to_ps of some other tensor with index j, then
    # (i, 1) and (j, 0) will be clustered together (and end up being mapped to the same indice in
    # the resulting einsum).
    # Note that this is a problem if two virtual dimensions (from possibly different
    # DiagonaSparseTensors) have the same size but not the same decomposition into physical
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
    tensors = list[Tensor]()
    indices_to_n_pdims = dict[int, int]()
    for t, indices in args:
        assert isinstance(t, DiagonalSparseTensor)
        tensors.append(t.physical)
        for ps, index in zip(t.v_to_ps, indices):
            if index in indices_to_n_pdims:
                assert indices_to_n_pdims[index] == len(ps)
            else:
                indices_to_n_pdims[index] = len(ps)
        p_to_vs = p_to_vs_from_v_to_ps(t.v_to_ps)
        for indices_ in p_to_vs:
            # elements in indices[indices_] map to the same dimension, they should be clustered
            # together
            group_indices([(indices[i], sub_i) for i, sub_i in indices_])
        # record the physical dimensions, index[v] for v in vs will end-up mapping to the same
        # final dimension as they were just clustered, so we can take the first, which exists as
        # t is a valid DST.
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

    physical = torch.einsum(*[x for y in zip(tensors, new_indices) for x in y], new_output)
    return DiagonalSparseTensor(physical, v_to_ps)


@DiagonalSparseTensor.implements(aten.bmm.default)
def bmm_default(mat1: Tensor, mat2: Tensor) -> DiagonalSparseTensor:
    assert isinstance(mat1, DiagonalSparseTensor) or isinstance(mat2, DiagonalSparseTensor)
    assert (
        mat1.ndim == 3
        and mat2.ndim == 3
        and mat1.shape[0] == mat2.shape[0]
        and mat1.shape[2] == mat2.shape[1]
    )

    mat1_ = to_diagonal_sparse_tensor(mat1)
    mat2_ = to_diagonal_sparse_tensor(mat2)

    # TODO: Verify that the dimension `0` of mat1_ and mat2_ have the same physical dimension sizes
    #  decompositions. If not, can reshape to common decomposition?
    return einsum((mat1_, [0, 1, 2]), (mat2_, [0, 2, 3]), output=[0, 1, 3])


@DiagonalSparseTensor.implements(aten.mm.default)
def mm_default(mat1: Tensor, mat2: Tensor) -> DiagonalSparseTensor:
    assert isinstance(mat1, DiagonalSparseTensor) or isinstance(mat2, DiagonalSparseTensor)
    assert mat1.ndim == 2 and mat2.ndim == 2 and mat1.shape[1] == mat2.shape[0]

    mat1_ = to_diagonal_sparse_tensor(mat1)
    mat2_ = to_diagonal_sparse_tensor(mat2)

    return einsum((mat1_, [0, 1]), (mat2_, [1, 2]), output=[0, 2])


# pointwise functions applied to one Tensor with `0.0 → 0`
_POINTWISE_FUNCTIONS = [
    aten.abs.default,
    aten.absolute.default,
    aten.asin.default,
    aten.asinh.default,
    aten.atan.default,
    aten.atanh.default,
    aten.ceil.default,
    aten.erf.default,
    aten.erfinv.default,
    aten.expm1.default,
    aten.fix.default,
    aten.floor.default,
    aten.hardtanh.default,
    aten.leaky_relu.default,
    aten.log1p.default,
    aten.neg.default,
    aten.negative.default,
    aten.positive.default,
    aten.relu.default,
    aten.round.default,
    aten.sgn.default,
    aten.sign.default,
    aten.sin.default,
    aten.sinh.default,
    aten.sqrt.default,
    aten.square.default,
    aten.tan.default,
    aten.tanh.default,
    aten.trunc.default,
]

_IN_PLACE_POINTWISE_FUNCTIONS = [
    aten.abs_.default,
    aten.absolute_.default,
    aten.asin_.default,
    aten.asinh_.default,
    aten.atan_.default,
    aten.atanh_.default,
    aten.ceil_.default,
    aten.erf_.default,
    aten.erfinv_.default,
    aten.expm1_.default,
    aten.fix_.default,
    aten.floor_.default,
    aten.hardtanh_.default,
    aten.leaky_relu_.default,
    aten.log1p_.default,
    aten.neg_.default,
    aten.negative_.default,
    aten.relu_.default,
    aten.round_.default,
    aten.sgn_.default,
    aten.sign_.default,
    aten.sin_.default,
    aten.sinh_.default,
    aten.sqrt_.default,
    aten.square_.default,
    aten.tan_.default,
    aten.tanh_.default,
    aten.trunc_.default,
]


def _override_pointwise(op):
    @DiagonalSparseTensor.implements(op)
    def func_(t: DiagonalSparseTensor) -> DiagonalSparseTensor:
        assert isinstance(t, DiagonalSparseTensor)
        return DiagonalSparseTensor(op(t.physical), t.v_to_ps)

    return func_


def _override_inplace_pointwise(op):
    @DiagonalSparseTensor.implements(op)
    def func_(t: DiagonalSparseTensor) -> DiagonalSparseTensor:
        assert isinstance(t, DiagonalSparseTensor)
        op(t.physical)
        return t


for pointwise_func in _POINTWISE_FUNCTIONS:
    _override_pointwise(pointwise_func)

for pointwise_func in _IN_PLACE_POINTWISE_FUNCTIONS:
    _override_inplace_pointwise(pointwise_func)
