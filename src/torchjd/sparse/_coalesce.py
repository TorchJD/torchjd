import torch
from torch import Tensor


def fix_zero_stride_columns(physical: Tensor, strides: Tensor) -> tuple[Tensor, Tensor]:
    """
    Remove columns of strides that are all 0 and sum the corresponding elements in the physical
    tensor.
    """

    are_columns_zero = (strides == 0).all(dim=0)

    if not are_columns_zero.any():
        return physical, strides

    zero_column_indices = torch.arange(len(are_columns_zero))[are_columns_zero].tolist()
    physical = physical.sum(dim=zero_column_indices)
    strides = strides[:, ~are_columns_zero]
    return physical, strides
