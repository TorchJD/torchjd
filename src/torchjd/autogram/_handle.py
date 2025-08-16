from torch.utils.hooks import RemovableHandle as TorchRemovableHandle


class RemovableHandle:
    """
    A handle which provides the capability to remove all hooks added by autogram.

    Typical usage is:

    >>> # Augment the model
    >>> handle = augment_model_for_iwrm(model, weighting)
    >>>
    >>>  # Use it
    >>>  # ...
    >>>
    >>> # De-augment the model
    >>> handle.remove()
    >>> # All hooks added by augment_model_for_iwrm have now been removed
    """

    def __init__(self, handles: list[TorchRemovableHandle]) -> None:
        self._handles = handles

    def remove(self):
        """
        Remove from a model and its submodules the module hooks added by autogram. This can be used
        to de-augment a model.
        """
        for handle in self._handles:
            handle.remove()
