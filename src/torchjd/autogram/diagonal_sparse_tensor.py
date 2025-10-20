import torch
from torch import Tensor
from torch.utils._pytree import tree_map


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
        result = Tensor._make_wrapper_subclass(cls, shape, dtype=data.dtype, device=data.device)
        result._data = data  # type: ignore
        result._v_to_p = v_to_p  # type: ignore
        result._v_shape = shape  # type: ignore

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

        output = torch.einsum(*einsum_args, output_indices)
        return output

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # TODO: Handle batched operations (apply to self._data and wrap)
        # TODO: Handle all operations that can be represented with an einsum by translating them
        #  to operations on self._data and wrapping accordingly.

        # --- Fallback: Fold to Dense Tensor ---
        def unwrap_to_dense(t):
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
