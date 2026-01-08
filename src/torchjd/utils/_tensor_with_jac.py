from torch import Tensor


class TensorWithJac(Tensor):
    """
    Tensor known to have a populated jac field.

    Should not be directly instantiated, but can be used as a type hint and can be casted to.
    """

    jac: Tensor
