from abc import ABC, abstractmethod

from torch.utils.hooks import RemovableHandle


class HandleManager(ABC):
    @abstractmethod
    def remove(self):
        """
        Remove handles from a model. This can be used to de-augment a model.
        """


class AutogramHandleManager(HandleManager):
    """
    Private `HandleManager` that is used to track Module hooks' handles to de-augment a model that
    was augmented for autogram.
    """

    def __init__(self) -> None:
        self._handles: list[RemovableHandle] = []

    def add_handle(self, handle: RemovableHandle) -> None:
        self._handles.append(handle)

    def remove(self):
        for handle in self._handles:
            handle.remove()
