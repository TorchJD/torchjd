# Minimal autograd + vmap-aware wrapper for TorchJD
# Input/Output RelationShip:  torch.sparse.mm(sparse[N,N], dense[N, d]) -> out[N, d]


from __future__ import annotations
import torch
from torch.autograd import Function
from torch.func import vmap # requires torch > 2.1



class _SparseMatMul(Function):
	@staticmethod
	def forward(ctx, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
		ctx.save_for_backward(sparse)
		return torch.sparse.mm(sparse, dense)


	@staticmethod
	def backward(ctx, grad_output: torch.Tensor):
		(sparse,) = ctx.saved_tensors
		grad_dense = torch.sparse.mm(                 # Aᵀ · g
	            sparse.transpose(0, 1), grad_output)
		return None, grad_dense


	@staticmethod
	def vmap(info, sparse_batched, dense_batched):
		sparse, = sparse_batched
		dense  = dense_batched
		B, N, d = dense.shape
		
		dense_2d = dense.reshape(B * N, d)
		dense_2d = dense_2d.view(N, B * d)

		out_2d = torch.sparse.mm(sparse, dense_2d)
		out    = out_2d.view(N, B, d).transpose(0, 1)

		return (out,)

        					
def sparse_mm(sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """
    vmap-compatible sparse @ dense.

    Example
    -------
    >>> out = sparse_mm(adj, feats)
    """
    return _SparseMatMul.apply(sparse, dense)
