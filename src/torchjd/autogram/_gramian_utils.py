from math import prod

from torch import Tensor


def reshape_gramian(gramian: Tensor, shape: list[int]) -> Tensor:
    """
    Reshapes a Gramian to a provided shape. As a Gramian is quadratic form, the reshape of the first
    half of the target dimensions must be done from the left, while the reshape of the second half
    must be done from the right.
    :param gramian: Gramian to reshape
    :param shape: First half of the target shape, the shape of the output is therefore
        `shape + shape[::-1]`.
    """

    # Example: `gramian` of shape [24, 24] and `shape` of [4, 3, 2]:
    # - The `unordered_gramian` will be of shape [4, 3, 2, 4, 3, 2]
    # - The `last_dims` will be [3, 4, 5] and `last_dims[::-1]` will be [5, 4, 3]
    # - The `reordered_gramian` will be of shape [4, 3, 2, 2, 3, 4]

    automatic_dimensions = [i for i in range(len(shape)) if shape[i] == -1]
    if len(automatic_dimensions) == 1:
        index = automatic_dimensions[0]
        current_shape = gramian.shape[: len(gramian.shape) // 2]
        numel = prod(current_shape)
        specified_numel = -prod(shape)  # shape[index] == -1, this is the product of all other dims
        shape[index] = numel // specified_numel

    unordered_intput_gramian = _revert_last_dims(gramian)
    unordered_output_gramian = unordered_intput_gramian.reshape(shape + shape)
    reordered_output_gramian = _revert_last_dims(unordered_output_gramian)
    return reordered_output_gramian


def movedim_gramian(gramian: Tensor, source: list[int], destination: list[int]) -> Tensor:
    """
    Moves the dimensions of a Gramian from some source dimensions to destination dimensions. As a
    Gramian is quadratic form, moving dimension must be done simultaneously on the first half of the
    dimensions and on the second half of the dimensions reversed.
    :param gramian: Gramian to reshape.
    :param source: Source dimensions, that should be in the range
        [-gramian.ndim//2, gramian.ndim//2[. Its elements should be unique.
    :param destination: Destination dimensions, that should be in the range
        [-gramian.ndim//2, gramian.ndim//2[. It should have the same size as `source`, and its
        elements should be unique.
    """

    # Example: `gramian` of shape [4, 3, 2, 2, 3, 4], `source` of [-2, 2] and destination of [0, 1]:
    # - `source_` will be [1, 2] and `destination_` will be [0, 1]
    # - `mirrored_source` will be [1, 2, 4, 3] and `mirrored_destination` will be [0, 1, 5, 4]
    # - The `moved_gramian` will be of shape [3, 2, 4, 4, 2, 3]

    # Map everything to the range [0, gramian.ndim//2[
    length = gramian.ndim // 2
    source_ = [i if 0 <= i else i + length for i in source]
    destination_ = [i if 0 <= i else i + length for i in destination]

    # Mirror the source and destination and use the result to move the dimensions of the gramian
    last_dim = gramian.ndim - 1
    mirrored_source = source_ + [last_dim - i for i in source_]
    mirrored_destination = destination_ + [last_dim - i for i in destination_]
    moved_gramian = gramian.movedim(mirrored_source, mirrored_destination)
    return moved_gramian


def _revert_last_dims(generalized_gramian: Tensor) -> Tensor:
    input_ndim = len(generalized_gramian.shape) // 2
    last_dims = [input_ndim + i for i in range(input_ndim)]
    return generalized_gramian.movedim(last_dims, last_dims[::-1])
