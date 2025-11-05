import itertools
import operator
from functools import wraps
from itertools import accumulate
from math import prod

import torch
from torch import Tensor, arange, meshgrid, stack, tensor, tensordot, zeros
from torch.utils._pytree import tree_map


class StructuredSparseTensor(Tensor):
    _HANDLED_FUNCTIONS = dict()

    @staticmethod
    def __new__(cls, physical: Tensor, strides: Tensor):
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

        pshape = torch.tensor(physical.shape)
        vshape = strides @ (pshape - 1) + 1
        return Tensor._make_wrapper_subclass(
            cls, vshape, dtype=physical.dtype, device=physical.device
        )

    def __init__(self, physical: Tensor, strides: Tensor):
        """
        This constructor is made for specifying physical and strides exactly. It should not modify
        it.

        For this reason, another constructor will be made to either modify the physical / strides to
        simplify the result, or to create a dense tensor directly if it's already dense.

        :param physical: The dense tensor holding the actual data.
        :param strides: Integer (int64) tensor of shape [virtual_ndim, physical_ndim], representing
            the linear transformation between an index in the physical tensor and the corresponding
            index in the virtual tensor, i.e. v_index = strides @ p_index.
        """

        if any(s == 1 for s in physical.shape):
            raise ValueError(
                "physical must not contain any dimension of size 1. Found physical.shape="
                f"{physical.shape}."
            )
        if strides.dtype is not torch.int64:
            raise ValueError(
                f"strides should be of int64 dtype. Found strides.dtype={strides.dtype}."
            )
        if not (strides >= 0).all():
            raise ValueError(f"All strides must be non-negative. Found strides={strides}.")
        if strides.shape[1] != physical.ndim:
            raise ValueError(
                f"strides should have 1 column per physical dimension. Found strides={strides} and physical.shape={physical.shape}."
            )
        if (strides.sum(dim=0) == 0).any():
            raise ValueError(
                f"strides should not have any column full of zeros. Found strides={strides}."
            )
        if any(len(group) != 1 for group in get_groupings(list(physical.shape), strides)):
            raise ValueError(f"Dimensions must be maximally grouped. Found strides={strides}.")

        self.physical = physical
        self.strides = strides

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
        return f"StructuredSparseTensor(physical={self.physical}, strides={self.strides})"

    def debug_info(self) -> str:
        info = (
            f"vshape: {self.shape}\n"
            f"pshape: {self.physical.shape}\n"
            f"strides: {self.strides}\n"
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


impl = StructuredSparseTensor.implements


def print_fallback(func, args, kwargs) -> None:
    def tensor_to_str(t: Tensor) -> str:
        result = f"{t.__class__.__name__} - vshape: {t.shape}"
        if isinstance(t, StructuredSparseTensor):
            result += f" - pshape: {t.physical.shape} - strides: {t.strides}"

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


def strides_v2(p_dims: list[int], physical_shape: list[int]) -> list[int]:
    """
    From a list of physical dimensions corresponding to a virtual dimension, and from the physical
    shape, get the stride indicating how moving on each physical dimension makes you move on the
    virtual dimension.

    Example:
        Imagine a vector of size 3, and of value [1, 2, 3].
        Imagine a SST t of shape [3, 3] using this vector as physical and using [[0, 0]] as v_to_ps.
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

    strides_v1 = list(accumulate([1] + [physical_shape[d] for d in p_dims[:0:-1]], operator.mul))[
        ::-1
    ]
    result = [0 for _ in range(len(physical_shape))]
    for i, d in enumerate(p_dims):
        result[d] += strides_v1[i]
    return result


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


def to_structured_sparse_tensor(t: Tensor) -> StructuredSparseTensor:
    if isinstance(t, StructuredSparseTensor):
        return t
    else:
        return make_sst(physical=t, strides=torch.eye(t.ndim, dtype=torch.int64))


def to_most_efficient_tensor(physical: Tensor, strides: Tensor) -> Tensor:
    physical, strides = fix_dim_of_size_1(physical, strides)
    physical, strides = fix_ungrouped_dims(physical, strides)

    if (strides.sum(dim=0) == 1).all():
        # All physical dimensions make you move by 1 in exactly 1 virtual dimension.
        # Also, because all physical dimensions have been maximally grouped, we cannot have two
        # physical dimensions that make you move in the same virtual dimension.
        # So strides is an identity matrix with potentially some extra rows of zeros, and
        # potentially shuffled columns.

        # The first step is to unsqueeze the physical tensor for each extra row of zeros in the
        # strides.
        zero_row_mask = strides.sum(dim=1) == 0
        number_of_zero_rows = zero_row_mask.sum()
        for _ in number_of_zero_rows:
            physical = physical.unsqueeze(-1)

        # The second step is to re-order the physical dimensions so that the corresponding
        # strides matrix would be an identity.
        source = arange(strides.shape[0])
        destination = strides[zero_row_mask] @ source
        return physical.movedim(list(source), list(destination))
    else:
        return StructuredSparseTensor(physical, strides)


def unwrap_to_dense(t: Tensor):
    if isinstance(t, StructuredSparseTensor):
        return t.to_dense()
    else:
        return t


def fix_dim_of_size_1(physical: Tensor, strides: Tensor) -> tuple[Tensor, Tensor]:
    is_of_size_1 = torch.tensor([s == 1 for s in physical.shape])
    return physical.squeeze(), strides[:, ~is_of_size_1]


def fix_ungrouped_dims(physical: Tensor, strides: Tensor) -> tuple[Tensor, Tensor]:
    groups = get_groupings(list(physical.shape), strides)
    nphysical = physical.reshape([prod([physical.shape[dim] for dim in group]) for group in groups])
    stride_mapping = torch.zeros(physical.ndim, nphysical.ndim, dtype=torch.int64)
    for j, group in enumerate(groups):
        stride_mapping[group[-1], j] = 1

    new_strides = strides @ stride_mapping
    return nphysical, new_strides


def make_sst(physical: Tensor, strides: Tensor) -> StructuredSparseTensor:
    """Fix physical and strides and create a StructuredSparseTensor with them."""

    physical, strides = fix_dim_of_size_1(physical, strides)
    physical, strides = fix_ungrouped_dims(physical, strides)
    return StructuredSparseTensor(physical, strides)


def fix_zero_stride_columns(physical: Tensor, strides: Tensor) -> tuple[Tensor, Tensor]:
    """Remove columns of strides that are all 0 and sum the corresponding elements in the physical tensor."""
    are_columns_zero = (strides == 0).all(dim=0)

    if not (are_columns_zero).any():
        return physical, strides

    zero_column_indices = torch.arange(len(are_columns_zero))[are_columns_zero].tolist()
    physical = physical.sum(dim=zero_column_indices)
    strides = strides[:, ~are_columns_zero]
    return physical, strides
