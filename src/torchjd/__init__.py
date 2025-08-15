"""
This package enable Jacobian descent, through the `backward` and `mtl_backward` functions, which
are meant to replace the call to `torch.backward` or `loss.backward` in gradient descent. To combine
the information of the Jacobian, an aggregator from the `aggregation` package has to be used.
"""

from warnings import warn as _warn

from .autojac import backward as _backward
from .autojac import mtl_backward as _mtl_backward

_deprecated_items = {
    "backward": ("autojac", _backward),
    "mtl_backward": ("autojac", _mtl_backward),
}


def __getattr__(name):
    """
    If an attribute is not found in the module's dictionary and its name is in _deprecated_items,
    then import it with a warning.
    """
    if name in _deprecated_items:
        _warn(
            f"Importing `{name}` from `torchjd` is deprecated. Please import it from "
            f"`{_deprecated_items[name][0]}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _deprecated_items[name][1]
    raise AttributeError(f"module {__name__} has no attribute {name}")
