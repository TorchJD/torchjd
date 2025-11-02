from torch import Tensor
from torch.ops import aten  # type: ignore

from torchjd.sparse import DiagonalSparseTensor


@DiagonalSparseTensor.implements(aten.threshold_backward.default)
def threshold_backward_default(
    grad_output: DiagonalSparseTensor, self: Tensor, threshold
) -> DiagonalSparseTensor:
    new_physical = aten.threshold_backward.default(grad_output.physical, self, threshold)

    return DiagonalSparseTensor(new_physical, grad_output.v_to_ps)


@DiagonalSparseTensor.implements(aten.hardtanh_backward.default)
def hardtanh_backward_default(
    grad_output: DiagonalSparseTensor,
    self: Tensor,
    min_val: Tensor | int | float,
    max_val: Tensor | int | float,
) -> DiagonalSparseTensor:
    if isinstance(self, DiagonalSparseTensor):
        raise NotImplementedError()

    new_physical = aten.hardtanh_backward.default(grad_output.physical, self, min_val, max_val)
    return DiagonalSparseTensor(new_physical, grad_output.v_to_ps)


@DiagonalSparseTensor.implements(aten.hardswish_backward.default)
def hardswish_backward_default(grad_output: DiagonalSparseTensor, self: Tensor):
    if isinstance(self, DiagonalSparseTensor):
        raise NotImplementedError()

    new_physical = aten.hardswish_backward.default(grad_output.physical, self)
    return DiagonalSparseTensor(new_physical, grad_output.v_to_ps)
