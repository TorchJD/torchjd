from torch import Tensor


def reshape_gramian(gramian: Tensor, half_shape: list[int]) -> Tensor:
    """
    Reshapes a Gramian to a provided shape. The reshape of the first half of the target dimensions
    must be done from the left, while the reshape of the second half must be done from the right.

    :param gramian: Gramian to reshape. Can be a generalized Gramian.
    :param half_shape: First half of the target shape, the shape of the output is therefore
        `shape + shape[::-1]`.
    """

    # Example 1: `gramian` of shape [4, 3, 2, 2, 3, 4] and `shape` of [8, 3]:
    # [4, 3, 2, 2, 3, 4] -(movedim)-> [4, 3, 2, 4, 3, 2] -(reshape)-> [8, 3, 8, 3] -(movedim)->
    # [8, 3, 3, 8]
    #
    # Example 2: `gramian` of shape [24, 24] and `shape` of [4, 3, 2]:
    # [24, 24] -(movedim)-> [24, 24] -(reshape)-> [4, 3, 2, 4, 3, 2] -(movedim)-> [4, 3, 2, 2, 3, 4]

    return _revert_last_dims(_revert_last_dims(gramian).reshape(half_shape + half_shape))


def _revert_last_dims(gramian: Tensor) -> Tensor:
    """Inverts the order of the last half of the dimensions of the input generalized Gramian."""

    half_ndim = gramian.ndim // 2
    last_dims = [half_ndim + i for i in range(half_ndim)]
    return gramian.movedim(last_dims, last_dims[::-1])


def movedim_gramian(gramian: Tensor, half_source: list[int], half_destination: list[int]) -> Tensor:
    """
    Moves the dimensions of a Gramian from some source dimensions to destination dimensions. This
    must be done simultaneously on the first half of the dimensions and on the second half of the
    dimensions reversed.

    :param gramian: Gramian to reshape. Can be a generalized Gramian.
    :param half_source: Source dimensions, that should be in the range [-gramian.ndim//2,
        gramian.ndim//2[. Its elements should be unique.
    :param half_destination: Destination dimensions, that should be in the range
        [-gramian.ndim//2, gramian.ndim//2[. It should have the same size as `half_source`, and its
        elements should be unique.
    """

    # Example: `gramian` of shape [4, 3, 2, 2, 3, 4], `half_source` of [-2, 2] and
    # `half_destination` of [0, 1]:
    # - `half_source_` will be [1, 2] and `half_destination_` will be [0, 1]
    # - `source` will be [1, 2, 4, 3] and `destination` will be [0, 1, 5, 4]
    # - The `moved_gramian` will be of shape [3, 2, 4, 4, 2, 3]

    # Map everything to the range [0, gramian.ndim//2[
    half_ndim = gramian.ndim // 2
    half_source_ = [i if 0 <= i else i + half_ndim for i in half_source]
    half_destination_ = [i if 0 <= i else i + half_ndim for i in half_destination]

    # Mirror the half source and the half destination and use the result to move the dimensions of
    # the gramian
    last_dim = gramian.ndim - 1
    source = half_source_ + [last_dim - i for i in half_source_]
    destination = half_destination_ + [last_dim - i for i in half_destination_]
    moved_gramian = gramian.movedim(source, destination)
    return moved_gramian
