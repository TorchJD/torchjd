from torch import Tensor


def _vector_to_str(vector: Tensor) -> str:
    """
    Transforms a Tensor of the form `tensor([1.23456, 1.0, ...])` into a string of the form
    `1.23, 1., ...`
    """

    weights_str = ", ".join(["{:.2f}".format(value).rstrip("0") for value in vector])
    return weights_str
