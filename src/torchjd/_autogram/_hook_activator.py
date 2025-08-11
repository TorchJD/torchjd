from collections.abc import Callable

from torch import nn
from torch.utils._pytree import PyTree


class HookActivator:
    """
    This class converts module hooks into hooks that can be activated or deactivated.
    """

    def __init__(self):
        self.is_active = True

    def activate(self) -> None:
        self.is_active = True

    def deactivate(self) -> None:
        self.is_active = False

    def convert_hook(
        self, hook: Callable[[nn.Module, PyTree, PyTree], PyTree]
    ) -> Callable[[nn.Module, PyTree, PyTree], PyTree]:
        def activated_hook(module: nn.Module, args: PyTree, output: PyTree):
            if not self.is_active:
                return output
            return hook(module, args, output)

        return activated_hook
