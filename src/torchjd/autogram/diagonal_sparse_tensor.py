from typing import Any

import torch
from torch import Tensor
from torch.ops import aten  # type: ignore[attr-defined]
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
    def __new__(cls, data: Tensor, v_to_p: list[int]):
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

        shape = [data.shape[i] for i in v_to_p]
        return Tensor._make_wrapper_subclass(cls, shape, dtype=data.dtype, device=data.device)

    def __init__(self, data: Tensor, v_to_p: list[int]):
        self.contiguous_data = data  # self.data cannot be used here.
        self.v_to_p = v_to_p

    def to_dense(self) -> Tensor:
        if self.contiguous_data.ndim == 0:
            return self.contiguous_data
        p_index_ranges = [
            torch.arange(s, device=self.contiguous_data.device) for s in self.contiguous_data.shape
        ]
        p_indices_grid = torch.meshgrid(*p_index_ranges, indexing="ij")
        v_indices_grid = tuple(p_indices_grid[i] for i in self.v_to_p)

        res = torch.zeros(
            self.shape, device=self.contiguous_data.device, dtype=self.contiguous_data.dtype
        )
        res[v_indices_grid] = self.contiguous_data
        return res

    @classmethod
    def __torch_dispatch__(cls, func: {__name__}, types: Any, args: tuple = (), kwargs: Any = None):
        kwargs = {} if kwargs is None else kwargs

        if func in _HANDLED_FUNCTIONS:
            return _HANDLED_FUNCTIONS[func](*args, **kwargs)

        # --- Fallback: Fold to Dense Tensor ---
        def unwrap_to_dense(t: Tensor):
            if isinstance(t, cls):
                return t.to_dense()
            else:
                return t

        print(f"Falling back to dense for {func.__name__}...")
        return func(*tree_map(unwrap_to_dense, args), **tree_map(unwrap_to_dense, kwargs))

    def __repr__(self):
        return (
            f"DiagonalSparseTensor(data={self.contiguous_data}, v_to_p_map={self.v_to_p}, shape="
            f"{self.shape})"
        )


def diagonal_sparse_tensor(data: Tensor, v_to_p: list[int]) -> Tensor:
    if not all(0 <= i < data.ndim for i in v_to_p):
        raise ValueError(f"Elements in v_to_p map to dimensions in data. Found {v_to_p}.")
    if len(set(v_to_p)) != data.ndim:
        raise ValueError("Every dimension in data must appear at least once in v_to_p.")
    if len(v_to_p) == data.ndim:
        return torch.movedim(data, (list(range(data.ndim))), v_to_p)
    else:
        return DiagonalSparseTensor(data, v_to_p)


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
    def func_(t: Tensor):
        assert isinstance(t, DiagonalSparseTensor)
        return diagonal_sparse_tensor(op(t.contiguous_data), t.v_to_p)

    return func_


def _override_inplace_pointwise(op):
    @implements(op)
    def func_(t: Tensor) -> Tensor:
        assert isinstance(t, DiagonalSparseTensor)
        op(t.contiguous_data)
        return t


for func in _POINTWISE_FUNCTIONS:
    _override_pointwise(func)

for func in _IN_PLACE_POINTWISE_FUNCTIONS:
    _override_inplace_pointwise(func)


@implements(aten.mean.default)
def mean(t: Tensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t.contiguous_data) / t.numel()


@implements(aten.sum.default)
def sum(t: Tensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t.contiguous_data)


@implements(aten.pow.Tensor_Scalar)
def pow_Tensor_Scalar(t: Tensor, exponent: float) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)

    if exponent <= 0:
        # Need to densify because we don't have pow(0, exponent) = 0
        return aten.pow.Tensor_Scalar(t.to_dense(), exponent)

    new_contiguous_data = aten.pow.Tensor_Scalar(t.contiguous_data, exponent)
    return diagonal_sparse_tensor(new_contiguous_data, t.v_to_p)


# Somehow there's no pow_.Tensor_Scalar and pow_.Scalar takes tensor and scalar.
@implements(aten.pow_.Scalar)
def pow__Scalar(t: Tensor, exponent: float) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)

    if exponent <= 0:
        # Need to densify because we don't have pow(0, exponent) = 0
        # Note sure if it's even possible to densify in-place, so let's just raise an error.
        raise ValueError(f"in-place pow with an exponent of {exponent} (<= 0) is not supported.")

    aten.pow_.Scalar(t.contiguous_data, exponent)
    return t
