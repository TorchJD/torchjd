import itertools
import operator
from functools import wraps
from itertools import accumulate
from math import prod

import torch
from torch import Tensor, arange, meshgrid, stack, tensor, tensordot, zeros
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

        self.physical = physical
        self.v_to_ps = v_to_ps

        # strides is of shape [v_ndim, p_ndim], such that v_index = strides @ p_index
        self.strides = get_strides(list(self.physical.shape), v_to_ps)

        if any(len(group) != 1 for group in get_groupings(list(self.physical.shape), self.strides)):
            raise ValueError(f"Dimensions must be maximally grouped. Found {v_to_ps}.")

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


def get_strides(pshape: list[int], v_to_ps: list[list[int]]) -> Tensor:
    strides = torch.tensor([strides_v2(pdims, pshape) for pdims in v_to_ps], dtype=torch.int64)

    # It's sometimes necessary to reshape: when v_to_ps contains 0 element for instance.
    return strides.reshape(len(v_to_ps), len(pshape))


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def strides_to_pdims(strides: Tensor, physical_shape: list[int]) -> list[int]:
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
    remaining_strides = strides.clone()
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


def get_groupings(pshape: list[int], strides: Tensor) -> list[list[int]]:
    strides_time_pshape = strides * tensor(pshape)
    groups = {i: {i} for i, column in enumerate(strides.T)}
    group_ids = [i for i in range(len(strides.T))]
    for i1, i2 in itertools.combinations(range(strides.shape[1]), 2):
        if torch.equal(strides[:, i1], strides_time_pshape[:, i2]):
            groups[group_ids[i1]].update(groups[group_ids[i2]])
            group_ids[i2] = group_ids[i1]

    new_columns = [sorted(groups[group_id]) for group_id in sorted(set(group_ids))]

    if len(new_columns) != len(pshape):
        print(f"Combined pshape with the following new columns: {new_columns}.")

    return new_columns


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
        next_physical_index = physical.ndim
        new_v_to_ps = []
        # Add as many dimensions of size 1 as there are pdims equal to [] in v_to_ps.
        # Create the corresponding new_v_to_ps.
        # E.g. if v_to_ps is [[0], [], [1]], new_v_to_ps is [[0], [2], [1]].
        for vdim, pdims in enumerate(v_to_ps):
            if len(pdims) == 0:
                physical = physical.unsqueeze(-1)
                new_v_to_ps.append([next_physical_index])
                next_physical_index += 1
            else:
                new_v_to_ps.append(pdims)

        return torch.movedim(
            physical, list(range(physical.ndim)), [pdims[0] for pdims in new_v_to_ps]
        )
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
    strides = get_strides(list(physical.shape), v_to_ps)
    groups = get_groupings(list(physical.shape), strides)
    nphysical = physical.reshape([prod([physical.shape[dim] for dim in group]) for group in groups])
    stride_mapping = torch.zeros(physical.ndim, nphysical.ndim, dtype=torch.int64)
    for j, group in enumerate(groups):
        stride_mapping[group[-1], j] = 1

    new_strides = strides @ stride_mapping
    new_v_to_ps = [strides_to_pdims(stride, list(nphysical.shape)) for stride in new_strides]
    return nphysical, new_v_to_ps


def make_dst(physical: Tensor, v_to_ps: list[list[int]]) -> DiagonalSparseTensor:
    """Fix physical and v_to_ps and create a DiagonalSparseTensor with them."""

    physical, v_to_ps = fix_dim_encoding(physical, v_to_ps)
    physical, v_to_ps = fix_dim_of_size_1(physical, v_to_ps)
    physical, v_to_ps = fix_ungrouped_dims(physical, v_to_ps)
    return DiagonalSparseTensor(physical, v_to_ps)
