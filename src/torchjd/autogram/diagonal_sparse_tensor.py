from typing import Any

import torch
from torch import Tensor
from torch.ops import aten
from torch.utils._pytree import tree_map

_HANDLED_FUNCTIONS = dict()
import functools


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
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
        self._data = data
        self._v_to_p = v_to_p
        self._v_shape = [data.shape[i] for i in v_to_p]

    def to_dense(self) -> Tensor:
        if self._data.ndim == 0:
            return self._data
        p_index_ranges = [torch.arange(s, device=self._data.device) for s in self._data.shape]
        p_indices_grid = torch.meshgrid(*p_index_ranges)
        v_indices_grid = [p_indices_grid[i] for i in self._v_to_p]

        res = torch.zeros(self.shape, device=self._data.device, dtype=self._data.dtype)
        res[v_indices_grid] = self._data
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
            f"DiagonalSparseTensor(data={self._data}, v_to_p_map={self._v_to_p}, shape="
            f"{self._v_shape})"
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
_POINTWISE_FUNCTIONS = {
    aten.abs.default,
    aten.absolute.default,
    aten.neg.default,
    aten.negative.default,
    aten.sign.default,
    aten.sgn.default,
    aten.square.default,
    aten.fix.default,
    aten.floor.default,
    aten.ceil.default,
    aten.trunc.default,
    aten.round.default,
    aten.positive.default,
    aten.expm1.default,
    aten.log1p.default,
    aten.sqrt.default,
    aten.sin.default,
    aten.tan.default,
    aten.sinh.default,
    aten.tanh.default,
    aten.asin.default,
    aten.atan.default,
    aten.asinh.default,
    aten.atanh.default,
    aten.erf.default,
    aten.erfinv.default,
    aten.relu.default,
    aten.hardtanh.default,
    aten.leaky_relu.default,
}
_IN_PLACE_POINTWISE_FUNCTIONS = {
    aten.abs_.default,
    aten.absolute_.default,
    aten.neg_.default,
    aten.negative_.default,
    aten.sign_.default,
    aten.sgn_.default,
    aten.square_.default,
    aten.fix_.default,
    aten.floor_.default,
    aten.ceil_.default,
    aten.trunc_.default,
    aten.round_.default,
    aten.positive_.default,
    aten.expm1_.default,
    aten.log1p_.default,
    aten.sqrt_.default,
    aten.sin_.default,
    aten.tan_.default,
    aten.sinh_.default,
    aten.tanh_.default,
    aten.asin_.default,
    aten.atan_.default,
    aten.asinh_.default,
    aten.atanh_.default,
    aten.erf_.default,
    aten.erfinv_.default,
    aten.relu_.default,
    aten.hardtanh_.default,
    aten.leaky_relu_.default,
}


for func in _POINTWISE_FUNCTIONS:

    @implements(func)
    def func(t: Tensor) -> Tensor:
        assert isinstance(t, DiagonalSparseTensor)
        return diagonal_sparse_tensor(func(t._data), t._v_to_p)


for func in _IN_PLACE_POINTWISE_FUNCTIONS:

    @implements(func)
    def func(t: Tensor) -> Tensor:
        assert isinstance(t, DiagonalSparseTensor)
        func(t._data)
        return t


@implements(aten.mean.default)
def mean(t: Tensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t._data) / t.numel()


@implements(aten.sum.default)
def sum(t: Tensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t._data)
