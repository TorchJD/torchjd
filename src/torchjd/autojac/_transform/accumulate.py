from torch import Tensor

from .base import Transform
from .tensor_dict import EmptyTensorDict, Gradients


class Accumulate(Transform[Gradients, EmptyTensorDict]):
    def __call__(self, gradients: Gradients) -> EmptyTensorDict:
        """
        Accumulates gradients with respect to keys in their ``.grad`` field.
        """

        for key in gradients.keys():
            _check_expects_grad(key)
            if hasattr(key, "grad") and key.grad is not None:
                key.grad += gradients[key]
            else:
                # We clone the value because we do not want subsequent accumulations to also affect
                # this value (in case it is still used outside). We do not detach from the
                # computation graph because the value can have grad_fn that we want to keep track of
                # (in case it was obtained via create_graph=True and a differentiable aggregator).
                key.grad = gradients[key].clone()

        return EmptyTensorDict()

    def check_keys(self, input_keys: set[Tensor]) -> set[Tensor]:
        return set()


def _check_expects_grad(tensor: Tensor) -> None:
    if not _expects_grad(tensor):
        raise ValueError(
            "Cannot populate the .grad field of a Tensor that does not satisfy:"
            "`tensor.requires_grad and (tensor.is_leaf or tensor.retains_grad)`."
        )


def _expects_grad(tensor: Tensor) -> bool:
    """
    Determines whether a Tensor expects its .grad attribute to be populated.
    See https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf for more information.
    """

    return tensor.requires_grad and (tensor.is_leaf or tensor.retains_grad)
