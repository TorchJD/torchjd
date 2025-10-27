import operator
from itertools import accumulate
from math import prod

import torch
from torch import Tensor
from torch.ops import aten  # type: ignore
from torch.utils._pytree import tree_map

_HANDLED_FUNCTIONS = dict()
from functools import wraps


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    @wraps(func)
    def decorator(func):
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class DiagonalSparseTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, data: Tensor, v_to_ps: list[list[int]]):
        # At the moment, this class is not compositional, so we assert
        # that the tensor we're wrapping is exactly a Tensor
        assert type(data) is Tensor

        # Note [Passing requires_grad=true tensors to subclasses]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calling _make_subclass directly in an autograd context is
        # never the right thing to do, as this will detach you from
        # the autograd graph.  You must create an autograd function
        # representing the "constructor" (NegativeView, in this case)
        # and call that instead.  This assert helps prevent direct usage
        # (which is bad!)
        assert not data.requires_grad or not torch.is_grad_enabled()

        shape = [prod(data.shape[i] for i in dims) for dims in v_to_ps]
        return Tensor._make_wrapper_subclass(cls, shape, dtype=data.dtype, device=data.device)

    def __init__(self, data: Tensor, v_to_ps: list[list[int]]):
        if not all(len(dims) > 0 for dims in v_to_ps):
            raise ValueError(f"All elements of v_to_ps must be non-empty lists. Found {v_to_ps}.")
        if not all(all(0 <= dim < data.ndim for dim in dims) for dims in v_to_ps):
            raise ValueError(
                f"Elements in v_to_ps must map to dimensions in data. Found {v_to_ps}."
            )
        if len(set.union(*[set(dims) for dims in v_to_ps])) != data.ndim:
            raise ValueError("Every dimension in data must appear at least once in v_to_ps.")

        self.contiguous_data = data  # self.data cannot be used here.
        self.v_to_ps = v_to_ps

        # This is a list of strides whose shape matches that of v_to_ps except that each element
        # is the stride factor of the index to get the right element for the corresponding virtual
        # dimension. Stride is the jump necessary to go from one element to the next one in the
        # specified dimension. For instance if the i'th element of v_to_ps is [0, 1, 2], then the
        # i'th element of _strides is [data.shape[1] * data.shape[2], data.shape[2], 1] and so, if
        # we index dimension i with j=j_0 * stride[0] + j_1 * stride[1] + j_2 * stride[2], which is
        # a unique decomposition, then this corresponds to indexing dimensions v_to_ps[i] at indices
        # [j_0, j_1, j_2]
        s = data.shape
        self._strides = [
            list(accumulate([1] + [s[dim] for dim in dims[:0:-1]], operator.mul))[::-1]
            for dims in v_to_ps
        ]

    def to_dense(
        self, dtype: torch.dtype | None = None, *, masked_grad: bool | None = None
    ) -> Tensor:
        assert dtype is None  # We may add support for this later
        assert masked_grad is None  # We may add support for this later

        if self.contiguous_data.ndim == 0:
            return self.contiguous_data
        p_index_ranges = [
            torch.arange(s, device=self.contiguous_data.device) for s in self.contiguous_data.shape
        ]
        p_indices_grid = torch.meshgrid(*p_index_ranges, indexing="ij")
        v_indices_grid = list[Tensor]()
        for stride, dims in zip(self._strides, self.v_to_ps):
            stride_ = torch.tensor(stride, device=self.contiguous_data.device, dtype=torch.int)
            v_indices_grid.append(
                torch.sum(torch.stack([p_indices_grid[d] for d in dims], dim=-1) * stride_, dim=-1)
            )
            # This is supposed to be a vector of shape d_1 * d_2 ...
            # whose elements are the coordinates 1 in p_indices_grad[d_1] times stride 1
            # plus coordinates 2 in p_indices_grad[d_2] times stride 2, etc...

        res = torch.zeros(
            self.shape, device=self.contiguous_data.device, dtype=self.contiguous_data.dtype
        )
        res[tuple(v_indices_grid)] = self.contiguous_data
        return res

    def p_to_vs(self) -> list[list[tuple[int, int]]]:
        """
        A physical dimension is mapped to a list of couples of the form
        (virtual_dim, sub_index_in_virtual_dim)
        """
        res = dict[int, list[tuple[int, int]]]()
        for v_dim, p_dims in enumerate(self.v_to_ps):
            for i, p_dim in enumerate(p_dims):
                if p_dim not in res:
                    res[p_dim] = [(v_dim, i)]
                else:
                    res[p_dim].append((v_dim, i))
        return [res[i] for i in range(len(res))]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func in _HANDLED_FUNCTIONS:
            return _HANDLED_FUNCTIONS[func](*args, **kwargs)

        # --- Fallback: Fold to Dense Tensor ---
        def unwrap_to_dense(t: Tensor):
            if isinstance(t, cls):
                return t.to_dense()
            else:
                return t

        print(f"Falling back to dense for {func.__name__} called with the following args:")
        for arg in args:
            print(arg)
        print("and the following kwargs:")
        for k, v in kwargs.items():
            print(f"{k}: {v}")
        return func(*tree_map(unwrap_to_dense, args), **tree_map(unwrap_to_dense, kwargs))

    def __repr__(self):
        return (
            f"DiagonalSparseTensor(data={self.contiguous_data}, v_to_ps_map={self.v_to_ps}, shape="
            f"{self.shape})"
        )

    def debug_info(self) -> str:
        info = (
            f"shape: {self.shape}\n"
            f"stride(): {self.stride()}\n"
            f"v_to_ps: {self.v_to_ps}\n"
            f"contiguous_data.shape: {self.contiguous_data.shape}\n"
            f"contiguous_data.stride(): {self.contiguous_data.stride()}"
        )
        return info


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
    @implements(op)
    def func_(t: DiagonalSparseTensor) -> DiagonalSparseTensor:
        assert isinstance(t, DiagonalSparseTensor)
        return DiagonalSparseTensor(op(t.contiguous_data), t.v_to_ps)

    return func_


def _override_inplace_pointwise(op):
    @implements(op)
    def func_(t: DiagonalSparseTensor) -> DiagonalSparseTensor:
        assert isinstance(t, DiagonalSparseTensor)
        op(t.contiguous_data)
        return t


for func in _POINTWISE_FUNCTIONS:
    _override_pointwise(func)

for func in _IN_PLACE_POINTWISE_FUNCTIONS:
    _override_inplace_pointwise(func)


@implements(aten.mean.default)
def mean_default(t: DiagonalSparseTensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t.contiguous_data) / t.numel()


@implements(aten.sum.default)
def sum_default(t: DiagonalSparseTensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t.contiguous_data)


@implements(aten.pow.Tensor_Scalar)
def pow_Tensor_Scalar(t: DiagonalSparseTensor, exponent: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    if exponent <= 0.0:
        # Need to densify because we don't have pow(0.0, exponent) = 0.0
        return aten.pow.Tensor_Scalar(t.to_dense(), exponent)

    new_contiguous_data = aten.pow.Tensor_Scalar(t.contiguous_data, exponent)
    return DiagonalSparseTensor(new_contiguous_data, t.v_to_ps)


# Somehow there's no pow_.Tensor_Scalar and pow_.Scalar takes tensor and scalar.
@implements(aten.pow_.Scalar)
def pow__Scalar(t: DiagonalSparseTensor, exponent: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    if exponent <= 0.0:
        # Need to densify because we don't have pow(0.0, exponent) = 0.0
        # Note sure if it's even possible to densify in-place, so let's just raise an error.
        raise ValueError(f"in-place pow with an exponent of {exponent} (<= 0) is not supported.")

    aten.pow_.Scalar(t.contiguous_data, exponent)
    return t


@implements(aten.unsqueeze.default)
def unsqueeze_default(t: DiagonalSparseTensor, dim: int) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)
    assert -t.ndim - 1 <= dim < t.ndim + 1

    if dim < 0:
        dim = t.ndim + dim + 1

    new_data = aten.unsqueeze.default(t.contiguous_data, -1)
    new_v_to_ps = [p for p in t.v_to_ps]  # Deepcopy the list to not modify the original v_to_ps
    new_v_to_ps.insert(dim, [new_data.ndim - 1])

    return DiagonalSparseTensor(new_data, new_v_to_ps)


@implements(aten.view.default)
def view_default(t: DiagonalSparseTensor, shape: list[int]) -> DiagonalSparseTensor:
    # TODO: add error message when error is raised
    # TODO: handle case where the contiguous_data has to be reshaped

    assert isinstance(t, DiagonalSparseTensor)

    if prod(shape) != t.numel():
        raise ValueError(f"shape '{shape}' is invalid for input of size {t.numel()}")

    new_v_to_ps = []
    idx = 0
    flat_v_to_ps = [dim for dims in t.v_to_ps for dim in dims]
    p_shape = t.contiguous_data.shape
    for s in shape:
        # Always add the first element of the group, before even entering the while.
        # This is because both s and t.contiguous_data.shape[flat_v_to_ps[idx]] could be equal to 1,
        # in which case the while will not even be entered but we still want to add the dimension to
        # the group. More generally, it's a bit arbitrary in which groups the dimension of length 1
        # are put, but it should rarely be an issue.

        group = [flat_v_to_ps[idx]]
        current_product = p_shape[flat_v_to_ps[idx]]
        idx += 1

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

    return DiagonalSparseTensor(t.contiguous_data, new_v_to_ps)


@implements(aten.expand.default)
def expand_default(t: DiagonalSparseTensor, sizes: list[int]) -> DiagonalSparseTensor:
    # note that sizes could also be just an int, or a torch.Size i think
    assert isinstance(t, DiagonalSparseTensor)
    assert isinstance(sizes, list)
    assert len(sizes) == t.ndim

    new_contiguous_data_shape = [-1] * t.contiguous_data.ndim

    for dim, (original_size, new_size) in enumerate(zip(t.shape, sizes)):
        if new_size != original_size:
            if original_size != 1:
                raise ValueError("Cannot yet expand dim whose size != 1.")

            if len(t.v_to_ps[dim]) != 1:
                raise ValueError(
                    "Cannot yet expand virtual dim corresponding to several physical dims"
                )

            physical_dim = t.v_to_ps[dim][0]

            # Verify that we don't have two virtual dims expanding the same physical dim differently
            previous_value = new_contiguous_data_shape[physical_dim]
            assert previous_value == -1 or previous_value == new_size

            new_contiguous_data_shape[physical_dim] = new_size

    new_contiguous_data = aten.expand.default(t.contiguous_data, new_contiguous_data_shape)

    # Not sure if it's safe to just provide v_to_ps as-is. I think we're supposed to copy it.
    return DiagonalSparseTensor(new_contiguous_data, t.v_to_ps)


@implements(aten.div.Scalar)
def div_Scalar(t: DiagonalSparseTensor, divisor: float) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    new_contiguous_data = aten.div.Scalar(t.contiguous_data, divisor)
    return DiagonalSparseTensor(new_contiguous_data, t.v_to_ps)


@implements(aten.slice.Tensor)
def slice_Tensor(
    t: DiagonalSparseTensor, dim: int, start: int | None, end: int | None, step: int = 1
) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    physical_dims = t.v_to_ps[dim]

    if len(physical_dims) != 1:
        raise ValueError("Cannot yet slice virtual dim corresponding to several physical dims.")

    physical_dim = physical_dims[0]

    new_contiguous_data = aten.slice.Tensor(t.contiguous_data, physical_dim, start, end, step)

    return DiagonalSparseTensor(new_contiguous_data, t.v_to_ps)


@implements(aten.mul.Tensor)
def mul_Tensor(t1: Tensor, t2: DiagonalSparseTensor) -> DiagonalSparseTensor:
    # Element-wise multiplication where t1 is dense and t2 is DST
    assert isinstance(t2, DiagonalSparseTensor)

    new_contiguous_data = aten.mul.Tensor(t1, t2.contiguous_data)
    return DiagonalSparseTensor(new_contiguous_data, t2.v_to_ps)


@implements(aten.transpose.int)
def transpose_int(t: DiagonalSparseTensor, dim0: int, dim1: int) -> DiagonalSparseTensor:
    assert isinstance(t, DiagonalSparseTensor)

    new_v_to_ps = [dims for dims in t.v_to_ps]
    new_v_to_ps[dim0] = t.v_to_ps[dim1]
    new_v_to_ps[dim1] = t.v_to_ps[dim0]

    return DiagonalSparseTensor(t.contiguous_data, new_v_to_ps)


def einsum(*args: tuple[Tensor, list[int]], output: list[int]) -> DiagonalSparseTensor:
    # TODO: Handle ellipsis
    # TODO: Should we take only DiagonalSparseTensors and leave the responsability to cast to the
    #  caller?

    # If we have an index v for some virtual dim whose corresponding v_to_ps is a non-trivial list
    # [p_1, ..., p_k], then we have to create fresh sub-indices for each dimension.
    # For this reason, an index is decomposed into sub-indices that are then independently
    # clustered.
    # So if an index i in args for some DiagonalSparseTensor corresponds to a v_to_ps [j, k, l], then
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
        if isinstance(t, DiagonalSparseTensor):
            tensors.append(t.contiguous_data)
            for ps, index in zip(t.v_to_ps, indices):
                if index in indices_to_n_pdims:
                    assert indices_to_n_pdims[index] == len(ps)
                else:
                    indices_to_n_pdims[index] = len(ps)
            p_to_vs = t.p_to_vs()
            for indices_ in p_to_vs:
                # elements in indices[indices_] map to the same dimension, they should be clustered
                # together
                group_indices([(indices[i], sub_i) for i, sub_i in indices_])
            # record the physical dimensions, index[v] for v in vs will end-up mapping to the same
            # final dimension as they were just clustered, so we can take the first, which exists as
            # t is a valid DST.
            new_indices_pair.append([(indices[vs[0][0]], vs[0][1]) for vs in p_to_vs])
        else:
            tensors.append(t)
            new_indices_pair.append([(i, 0) for i in indices])

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

    data = torch.einsum(*[x for y in zip(tensors, new_indices) for x in y], new_output)
    return DiagonalSparseTensor(data, v_to_ps)


@implements(aten.bmm.default)
def bmm_default(mat1: Tensor, mat2: Tensor) -> Tensor:
    assert isinstance(mat1, DiagonalSparseTensor) or isinstance(mat2, DiagonalSparseTensor)
    assert (
        mat1.ndim == 3
        and mat2.ndim == 3
        and mat1.shape[0] == mat2.shape[0]
        and mat1.shape[2] == mat2.shape[1]
    )

    # TODO: Verify that if mat1 and/or mat2 are DiagonalSparseTensors, then their dimension `0` have
    #  the same physical dimension sizes decompositions.
    #  If not, can reshape to common decomposition?
    return einsum((mat1, [0, 1, 2]), (mat2, [0, 2, 3]), output=[0, 1, 3])
