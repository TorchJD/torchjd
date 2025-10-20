from torch import Tensor
from torch.ops import aten  # type: ignore

from torchjd.sparse._structured_sparse_tensor import StructuredSparseTensor, impl


@impl(aten.threshold_backward.default)
def threshold_backward_default(
    grad_output: StructuredSparseTensor, self: Tensor, threshold
) -> StructuredSparseTensor:
    new_physical = aten.threshold_backward.default(grad_output.physical, self, threshold)

    return StructuredSparseTensor(new_physical, grad_output.strides)


@impl(aten.hardtanh_backward.default)
def hardtanh_backward_default(
    grad_output: StructuredSparseTensor,
    self: Tensor,
    min_val: Tensor | int | float,
    max_val: Tensor | int | float,
) -> StructuredSparseTensor:
    if isinstance(self, StructuredSparseTensor):
        raise NotImplementedError()

    new_physical = aten.hardtanh_backward.default(grad_output.physical, self, min_val, max_val)
    return StructuredSparseTensor(new_physical, grad_output.strides)


@impl(aten.hardswish_backward.default)
def hardswish_backward_default(grad_output: StructuredSparseTensor, self: Tensor):
    if isinstance(self, StructuredSparseTensor):
        raise NotImplementedError()

    new_physical = aten.hardswish_backward.default(grad_output.physical, self)
    return StructuredSparseTensor(new_physical, grad_output.strides)
