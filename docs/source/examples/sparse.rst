Quick example
==============================

TorchJD now offers helpers that make working with sparse adjacency matrices
transparent.
The key entry-point is :pyfunc:`torchjd.sparse.sparse_mm`,
a vmap-aware autograd function that replaces the usual
``torch.sparse.mm`` inside Jacobian Descent pipelines.

The snippet below shows how you can mix a sparse objective (involving
``A @ p``) with a dense one, then aggregate their Jacobians using
:pyclass:`torchjd.aggregation.UPGrad`.

.. doctest::

    >>> import torch
    >>> from torchjd import backward
    >>> from torchjd.sparse import sparse_mm        # patches torch automatically
    >>> from torchjd.aggregation import UPGrad
    >>>
    >>> # 2×2 off-diagonal adjacency matrix
    >>> A = torch.sparse_coo_tensor(
    ...     indices=[[0, 1], [1, 0]],
    ...     values=[1.0, 1.0],
    ...     size=(2, 2)
    ... ).coalesce()
    >>>
    >>> p = torch.tensor([1.0, 2.0], requires_grad=True)
    >>>
    >>> y1 = sparse_mm(A, p.unsqueeze(1)).sum()   # sparse term
    >>> y2 = (p ** 2).sum()                      # dense term
    >>> backward([y1, y2], UPGrad())             # Jacobian Descent step
    >>> p.grad                                    # doctest:+ELLIPSIS
    tensor([1.0000, 1.6667])
