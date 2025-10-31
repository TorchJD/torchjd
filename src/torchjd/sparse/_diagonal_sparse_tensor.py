import operator
from functools import wraps
from itertools import accumulate
from math import prod

import torch
from torch import Tensor, arange, meshgrid, stack, tensor, tensordot, zeros
from torch.ops import aten  # type: ignore
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


class DiagonalSparseTensor(Tensor):
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
        pshape = list(self.physical.shape)
        self.strides = tensor([strides_v2(pdims, pshape) for pdims in self.v_to_ps])

    def to_dense(
        self, dtype: torch.dtype | None = None, *, masked_grad: bool | None = None
    ) -> Tensor:
        assert dtype is None  # We may add support for this later
        assert masked_grad is None  # We may add support for this later

        if self.physical.ndim == 0:
            return self.physical

        p_index_ranges = [arange(s) for s in self.physical.shape]
        p_indices_grid = stack(meshgrid(*p_index_ranges, indexing="ij"))

        # addmm_cuda not implemented for Long tensors => gotta have these tensors on cpu
        v_indices_grid = tensordot(self.strides, p_indices_grid, dims=1)
        res = zeros(self.shape, device=self.device, dtype=self.dtype)
        res[tuple(v_indices_grid)] = self.physical
        return res

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in cls._HANDLED_FUNCTIONS:
            return cls._HANDLED_FUNCTIONS[func](*args, **kwargs)

        print_fallback(func, args, kwargs)
        unwrapped_args = tree_map(unwrap_to_dense, args)
        unwrapped_kwargs = tree_map(unwrap_to_dense, kwargs)
        return func(*unwrapped_args, **unwrapped_kwargs)

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"DiagonalSparseTensor(physical={self.physical}, v_to_ps={self.v_to_ps})"

    def debug_info(self) -> str:
        info = (
            f"shape: {self.shape}\n"
            f"stride(): {self.stride()}\n"
            f"v_to_ps: {self.v_to_ps}\n"
            f"strides: {self.strides}\n"
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


def print_fallback(func, args, kwargs) -> None:
    def tensor_to_str(t: Tensor) -> str:
        result = f"{t.__class__.__name__} - shape: {t.shape}"
        if isinstance(t, DiagonalSparseTensor):
            result += f" - pshape: {t.physical.shape} - v_to_ps: {t.v_to_ps}"

        return result

    print(f"Falling back to dense for {func.__name__}")
    if len(args) > 0:
        print("* args:")
        for arg in args:
            if isinstance(arg, Tensor):
                print(f"  > {tensor_to_str(arg)}")
            elif isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], Tensor):
                list_content = "\n     ".join([tensor_to_str(t) for t in arg])
                print(f"  > [{list_content}]")
            else:
                print(f"  > {arg}")
    if len(kwargs) > 0:
        print("* kwargs:")
        for k, v in kwargs.items():
            print(f"  > {k}: {v}")
    print()


def strides_from_p_dims_and_p_shape(p_dims: list[int], physical_shape: list[int]) -> list[int]:
    return list(accumulate([1] + [physical_shape[dim] for dim in p_dims[:0:-1]], operator.mul))[
        ::-1
    ]


def strides_v2(p_dims: list[int], physical_shape: list[int]) -> list[int]:
    """
    From a list of physical dimensions corresponding to a virtual dimension, and from the physical
    shape, get the stride indicating how moving on each physical dimension makes you move on the
    virtual dimension.

    Example:
        Imagine a vector of size 3, and of value [1, 2, 3].
        Imagine a DST t of shape [3, 3] using this vector as physical and using [[0, 0]] as v_to_ps.
        t.to_dense() is [1, 0, 0, 0, 2, 0, 0, 0, 3] (it's the flattening of the diagonal matrix
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]]).
        When you move by 1 on physical dimension 0, you move by 4 on virtual dimension 0, i.e.
        strides_v2([0, 0], [3]) = 4
        In the 2D view, you'd move by 1 row (3 indices) and 1 column (1 index).

    Example:
        strides_v2([0, 0, 1], [3,4])  # [16, 1]
        Moving by 1 on physical dimension 0 makes you move by 16 on the virtual dimension. Moving by
        1 on physical dimension 1 makes you move by 1 on the virtual dimension.
    """

    strides_v1 = strides_from_p_dims_and_p_shape(p_dims, physical_shape)
    result = [0 for _ in range(len(physical_shape))]
    for i, d in enumerate(p_dims):
        result[d] += strides_v1[i]
    return result


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def strides_to_pdims(strides: list[int], physical_shape: list[int]) -> list[int]:
    """
    Given a list of strides, find and return the used physical dimensions.

    This algorithm runs in O(n * m) with n the number of physical dimensions (i.e.
    len(physical_shape) and len(strides)), and with m the number of pdims in the result.

    I'm pretty sure it could be implemented in O((n+m)log(n)) by using a sorted linked list for the
    remaining_strides, and keeping it sorted each time we update it. Argmax would then always be 0,
    removing the need to go through the whole list at every iteration.
    """

    # e.g. strides = [22111, 201000], physical_shape = [10, 2]

    pdims = []
    remaining_strides = [s for s in strides]
    remaining_numel = (
        sum(remaining_strides[i] * (physical_shape[i] - 1) for i in range(len(physical_shape))) + 1
    )
    # e.g. 9 * 22111 + 1 * 201000 + 1 = 400000

    while sum(remaining_strides) > 0:
        current_pdim = argmax(remaining_strides)
        # e.g. 1

        pdims.append(current_pdim)

        remaining_numel = remaining_numel // physical_shape[current_pdim]
        # e.g. 400000 / 2 = 200000

        remaining_strides[current_pdim] -= remaining_numel
        # e.g. [22111, 1000]

    return pdims


def merge_strides(strides: list[list[int]]) -> list[int]:
    return sorted({s for stride in strides for s in stride}, reverse=True)


def stride_to_shape(numel: int, stride: list[int]) -> list[int]:
    augmented_stride = [numel] + stride
    return [a // b for a, b in zip(augmented_stride[:-1], augmented_stride[1:])]


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

        available_dims = set(group) - visited_is
        groups.append(list(available_dims))
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


def to_most_efficient_tensor(physical: Tensor, v_to_ps: list[list[int]]) -> Tensor:
    physical, v_to_ps = fix_dim_encoding(physical, v_to_ps)
    physical, v_to_ps = fix_dim_of_size_1(physical, v_to_ps)
    physical, v_to_ps = fix_ungrouped_dims(physical, v_to_ps)

    if sum([len(pdims) for pdims in v_to_ps]) == physical.ndim:
        return torch.movedim(physical, list(range(physical.ndim)), [pdims[0] for pdims in v_to_ps])
    else:
        return DiagonalSparseTensor(physical, v_to_ps)


def unwrap_to_dense(t: Tensor):
    if isinstance(t, DiagonalSparseTensor):
        return t.to_dense()
    else:
        return t


def to_target_physical_strides(
    physical: Tensor, v_to_ps: list[list[int]], strides: list[list[int]]
) -> tuple[Tensor, list[list[int]]]:
    current_strides = [
        strides_from_p_dims_and_p_shape(p_dims, list(physical.shape)) for p_dims in v_to_ps
    ]
    target_stride = merge_strides(strides)

    numel = physical.numel()
    target_shape = stride_to_shape(numel, target_stride)
    new_physical = physical.reshape(target_shape)

    stride_to_p_dim = {s: i for i, s in enumerate(target_stride)}
    stride_to_p_dim[0] = len(target_shape)

    new_v_to_ps = list[list[int]]()
    for stride in current_strides:
        extended_stride = stride + [0]
        new_p_dims = list[int]()
        for s_curr, s_next in zip(extended_stride[:-1], extended_stride[1:]):
            new_p_dims += range(stride_to_p_dim[s_curr], stride_to_p_dim[s_next])
        new_v_to_ps.append(new_p_dims)

    return new_physical, new_v_to_ps


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


@DiagonalSparseTensor.implements(aten.sum.dim_IntList)
def sum_dim_IntList(
    t: DiagonalSparseTensor, dim: list[int], keepdim: bool = False, dtype=None
) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)

    if dtype:
        raise NotImplementedError()

    all_dims = list(range(t.ndim))
    result = einsum((t, all_dims), output=[d for d in all_dims if d not in dim])

    if keepdim:
        for d in dim:
            result = result.unsqueeze(d)

    return result


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


@DiagonalSparseTensor.implements(aten.permute.default)
def permute_default(t: DiagonalSparseTensor, dims: list[int]) -> DiagonalSparseTensor:
    new_v_to_ps = [t.v_to_ps[d] for d in dims]

    new_physical, new_v_to_ps = fix_dim_encoding(t.physical, new_v_to_ps)
    return DiagonalSparseTensor(new_physical, new_v_to_ps)


@DiagonalSparseTensor.implements(aten.cat.default)
def cat_default(tensors: list[Tensor], dim: int) -> Tensor:
    if any(not isinstance(t, DiagonalSparseTensor) for t in tensors):
        print_fallback(aten.cat.default, (tensors, dim), {})
        return aten.cat.default([unwrap_to_dense(t) for t in tensors])

    else:
        # TODO: efficient implementation when all tensors are sparse
        return aten.cat.default([unwrap_to_dense(t) for t in tensors])


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


def infer_shape(shape: list[int], numel: int) -> list[int]:
    if shape.count(-1) > 1:
        raise ValueError("Only one dimension can be inferred")
    known = 1
    for s in shape:
        if s != -1:
            known *= s
    inferred = numel // known
    return [inferred if s == -1 else s for s in shape]


@DiagonalSparseTensor.implements(aten.view.default)
def view_default(t: DiagonalSparseTensor, shape: list[int]) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)

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


@DiagonalSparseTensor.implements(aten._unsafe_view.default)
def _unsafe_view_default(t: DiagonalSparseTensor, shape: list[int]) -> Tensor:
    return view_default(
        t, shape
    )  # We don't do the optimizations that they do in https://github.com/pytorch/pytorch/blame/main/aten/src/ATen/native/TensorShape.cpp


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


@DiagonalSparseTensor.implements(aten.broadcast_tensors.default)
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


@DiagonalSparseTensor.implements(aten.div.Scalar)
def div_Scalar(t: DiagonalSparseTensor, divisor: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    new_physical = aten.div.Scalar(t.physical, divisor)
    return DiagonalSparseTensor(new_physical, t.v_to_ps)


@DiagonalSparseTensor.implements(aten.threshold_backward.default)
def threshold_backward_default(
    grad_output: DiagonalSparseTensor, self: Tensor, threshold
) -> DiagonalSparseTensor:
    new_physical = aten.threshold_backward.default(grad_output.physical, self, threshold)

    return DiagonalSparseTensor(new_physical, grad_output.v_to_ps)


@DiagonalSparseTensor.implements(aten.slice.Tensor)
def slice_Tensor(
    t: DiagonalSparseTensor, dim: int, start: int | None, end: int | None, step: int = 1
) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    physical_dims = t.v_to_ps[dim]

    if len(physical_dims) > 1:
        raise ValueError(
            "Cannot yet slice virtual dim corresponding to several physical dims.\n"
            f"{t.debug_info()}\n"
            f"dim={dim}, start={start}, end={end}, step={step}."
        )
    elif len(physical_dims) == 0:
        # Trying to slice a virtual dim of size 1.
        # Either
        # - the element of this dim is included in the slice: keep it as it is
        # - it's not included in the slice (e.g. end<=start): we would end up with a size of 0 on
        #   that dimension, so we'd need to add a dimension of size 0 to the physical. This is not
        #   implemented yet.
        start_ = start if start is not None else 0
        end_ = end if end is not None else 1
        if end_ <= start_:  # TODO: the condition might be a bit more complex if step != 1
            raise NotImplementedError(
                "Slicing of dimension of size 1 leading to dimension of size 0 not implemented yet."
            )
        else:
            new_physical = t.physical
    else:
        physical_dim = physical_dims[0]
        new_physical = aten.slice.Tensor(t.physical, physical_dim, start, end, step)

    return DiagonalSparseTensor(new_physical, t.v_to_ps)


@DiagonalSparseTensor.implements(aten.mul.Tensor)
def mul_Tensor(t1: Tensor | int | float, t2: Tensor | int | float) -> DiagonalSparseTensor:
    # Element-wise multiplication with broadcasting
    assert isinstance(t1, DiagonalSparseTensor) or isinstance(t2, DiagonalSparseTensor)

    if isinstance(t1, int) or isinstance(t1, float):
        t1_ = tensor(t1, device=t2.device)
    else:
        t1_ = t1

    if isinstance(t2, int) or isinstance(t2, float):
        t2_ = tensor(t2, device=t1.device)
    else:
        t2_ = t2

    t1_, t2_ = aten.broadcast_tensors.default([t1_, t2_])
    t1_ = to_diagonal_sparse_tensor(t1_)
    t2_ = to_diagonal_sparse_tensor(t2_)

    all_dims = list(range(t1_.ndim))
    return einsum((t1_, all_dims), (t2_, all_dims), output=all_dims)


@DiagonalSparseTensor.implements(aten.mul.Scalar)
def mul_Scalar(t: DiagonalSparseTensor, scalar) -> DiagonalSparseTensor:
    # TODO: maybe it could be that scalar is a scalar DST and t is a normal tensor. Need to check
    #  that

    assert isinstance(t, DiagonalSparseTensor)
    new_physical = aten.mul.Scalar(t.physical, scalar)
    return DiagonalSparseTensor(new_physical, t.v_to_ps)


@DiagonalSparseTensor.implements(aten.transpose.int)
def transpose_int(t: DiagonalSparseTensor, dim0: int, dim1: int) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    new_v_to_ps = [dims for dims in t.v_to_ps]
    new_v_to_ps[dim0] = t.v_to_ps[dim1]
    new_v_to_ps[dim1] = t.v_to_ps[dim0]

    new_physical, new_v_to_ps = fix_dim_encoding(t.physical, new_v_to_ps)
    return DiagonalSparseTensor(new_physical, new_v_to_ps)


def einsum(*args: tuple[DiagonalSparseTensor, list[int]], output: list[int]) -> Tensor:

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
    # build resulting dst

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
    # Need to use the safe constructor, otherwise the dimensions may not be maximally grouped.
    # Maybe there is a way to fix that though.
    return to_most_efficient_tensor(physical, v_to_ps)


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
