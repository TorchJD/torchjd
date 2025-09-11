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

    # Example 1: `gramian` of shape [4, 3, 2, 2, 3, 4] and `shape` of [8, 3]:
    # [4, 3, 2, 2, 3, 4] -(movedim)-> [4, 3, 2, 4, 3, 2] -(reshape)-> [8, 3, 8, 3] -(movedim)->
    # [8, 3, 3, 8]
    #
    # Example 2: `gramian` of shape [24, 24] and `shape` of [4, 3, 2]:
    # [24, 24] -(movedim)-> [24, 24] -(reshape)-> [4, 3, 2, 4, 3, 2] -(movedim)-> [4, 3, 2, 2, 3, 4]

    return _revert_last_dims(_revert_last_dims(gramian).reshape(shape + shape))


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
