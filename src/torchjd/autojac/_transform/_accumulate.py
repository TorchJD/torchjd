from torch import Tensor

from torchjd.utils._accumulation import _accumulate_grads, _accumulate_jacs

from ._base import TensorDict, Transform


class AccumulateGrad(Transform):
    """
    Transform from Gradients to {} that accumulates gradients with respect to keys into their
    ``grad`` field.
    """

    def __call__(self, gradients: TensorDict) -> TensorDict:
        _accumulate_grads(gradients.keys(), gradients.values())
        return {}

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        return set()


class AccumulateJac(Transform):
    """
    Transform from Jacobians to {} that accumulates jacobians with respect to keys into their
    ``jac`` field.
    """

    def __call__(self, jacobians: TensorDict) -> TensorDict:
        _accumulate_jacs(jacobians.keys(), jacobians.values())
        return {}

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        return set()
