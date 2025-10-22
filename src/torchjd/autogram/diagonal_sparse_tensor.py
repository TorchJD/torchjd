from typing import Any

import torch
from torch import Tensor
from torch.ops import aten
from torch.utils._pytree import tree_map

# pointwise functions applied to one Tensor with `0.0 → 0`
_POINTWISE_FUNCTIONS = {
    aten.abs.default,
    aten.abs_.default,
    aten.absolute.default,
    aten.absolute_.default,
    aten.neg.default,
    aten.neg_.default,
    aten.negative.default,
    aten.negative_.default,
    aten.sign.default,
    aten.sign_.default,
    aten.sgn.default,
    aten.sgn_.default,
    aten.square.default,
    aten.square_.default,
    aten.fix.default,
    aten.fix_.default,
    aten.floor.default,
    aten.floor_.default,
    aten.ceil.default,
    aten.ceil_.default,
    aten.trunc.default,
    aten.trunc_.default,
    aten.round.default,
    aten.round_.default,
    aten.positive.default,
    aten.expm1.default,
    aten.expm1_.default,
    aten.log1p.default,
    aten.log1p_.default,
    aten.sqrt.default,
    aten.sqrt_.default,
    aten.sin.default,
    aten.sin_.default,
    aten.tan.default,
    aten.tan_.default,
    aten.sinh.default,
    aten.sinh_.default,
    aten.tanh.default,
    aten.tanh_.default,
    aten.asin.default,
    aten.asin_.default,
    aten.atan.default,
    aten.atan_.default,
    aten.asinh.default,
    aten.asinh_.default,
    aten.atanh.default,
    aten.atanh_.default,
    aten.erf.default,
    aten.erf_.default,
    aten.erfinv.default,
    aten.erfinv_.default,
    aten.relu.default,
    aten.relu_.default,
    aten.hardtanh.default,
    aten.hardtanh_.default,
    aten.leaky_relu.default,
    aten.leaky_relu_.default,
}
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

        # TODO: assert a minimal data, all of its dimensions must be used at least once
        # TODO: If no repeat in v_to_p, return a view of data (non sparse tensor). If this cannot be
        #  done in __new__, create a helper function for that, and use this one everywhere.

        shape = [data.shape[i] for i in v_to_p]
        return Tensor._make_wrapper_subclass(cls, shape, dtype=data.dtype, device=data.device)

    def __init__(self, data: Tensor, v_to_p: list[int]):
        self._data = data
        self._v_to_p = v_to_p
        self._v_shape = [data.shape[i] for i in v_to_p]

    def to_dense(self) -> Tensor:
        first_indices = dict[int, int]()
        identity_matrices = dict[int, Tensor]()
        einsum_args: list[Tensor | list[int]] = [self._data, list(range(self._data.ndim))]
        output_indices = list(range(len(self._v_to_p)))
        for i, j in enumerate(self._v_to_p):
            if j not in first_indices:
                first_indices[j] = i
            else:
                if j not in identity_matrices:
                    device = self._data.device
                    dtype = self._data.dtype
                    identity_matrices[j] = torch.eye(self._v_shape[i], device=device, dtype=dtype)
                einsum_args += [identity_matrices[j], [first_indices[j], i]]

        # Need to be careful about nans, we would want to get identity times nan.
        output = torch.einsum(*einsum_args, output_indices)
        return output

    @classmethod
    def __torch_dispatch__(cls, func: {__name__}, types: Any, args: tuple = (), kwargs: Any = None):
        kwargs = {} if kwargs is None else kwargs

        # If `func` is a pointwise operator that applies to a single Tensor and such that func(0)=0
        # Then we can apply the transformation to self._data and wrap the result.
        if func in _POINTWISE_FUNCTIONS:
            assert (
                isinstance(args, tuple) and len(args) == 1 and func(torch.zeros([])).item() == 0.0
            )
            sparse_tensor = args[0]
            assert isinstance(sparse_tensor, DiagonalSparseTensor)
            new_data = func(sparse_tensor._data)
            return DiagonalSparseTensor(new_data, sparse_tensor._v_to_p)

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


@implements(aten.mean.default)
def mean_default(t: Tensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t._data) / t.numel()


@implements(aten.sum.default)
def sum_default(t: Tensor) -> Tensor:
    assert isinstance(t, DiagonalSparseTensor)
    return aten.sum.default(t._data)
