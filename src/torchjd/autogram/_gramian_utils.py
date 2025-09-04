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
    # - The returned gramian will be of shape [4, 3, 2, 2, 3, 4]

    target_ndim = len(shape)
    unordered_shape = shape + shape
    unordered_gramian = gramian.reshape(unordered_shape)
    last_dims = [target_ndim + i for i in range(target_ndim)]
    return unordered_gramian.movedim(last_dims, last_dims[::-1])


def movedim_gramian(gramian: Tensor, source: list[int], destination: list[int]) -> Tensor:
    """
    Moves the dimensions of a Gramian from some source dimensions to destination dimensions. As a
    Gramian is quadratic form, moving dimension must be done simultaneously on the first half of the
    dimensions and on the second half of the dimensions reversed.
    :param gramian: Gramian to reshape.
    :param source: Source dimensions, should be in the range [0, gramian.ndim/2]. Should be unique
    :param destination: Destination dimensions, should be in the range [0, gramian.ndim/2]. Should
        be unique and should have the same size as `source`.
    """

    length = gramian.ndim // 2
    source = [i if 0 <= i else i + length for i in source]
    destination = [i if 0 <= i else i + length for i in destination]

    last_index = gramian.ndim - 1
    source_dims = source + [last_index - i for i in source]
    destination_dims = destination + [last_index - i for i in destination]
    return gramian.movedim(source_dims, destination_dims)
