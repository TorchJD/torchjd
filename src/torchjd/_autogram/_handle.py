from torch.utils.hooks import RemovableHandle


class RemovableHandle:
    """TODO: add docstring (user-facing)"""

    def __init__(self, handles: list[RemovableHandle]) -> None:
        self._handles = handles

    def remove(self):
        for handle in self._handles:
            handle.remove()
