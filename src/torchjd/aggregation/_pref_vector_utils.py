from torch import Tensor

from ._str_utils import _vector_to_str
from .bases import _Weighting
from .constant import _ConstantWeighting


def _check_pref_vector(pref_vector: Tensor | None) -> None:
    """Checks the correctness of the parameter pref_vector."""

    if pref_vector is not None:
        if pref_vector.ndim != 1:
            raise ValueError(
                "Parameter `pref_vector` must be a vector (1D Tensor). Found `pref_vector.ndim = "
                f"{pref_vector.ndim}`."
            )


def _pref_vector_to_weighting(pref_vector: Tensor | None, default: _Weighting) -> _Weighting:
    """
    Returns the weighting associated to a given preference vector, with a fallback to a default
    weighting if the preference vector is None.
    """

    if pref_vector is None:
        return default
    else:
        return _ConstantWeighting(pref_vector)


def _pref_vector_to_str_suffix(pref_vector: Tensor | None) -> str:
    """Returns a suffix string containing the representation of the optional preference vector."""

    if pref_vector is None:
        return ""
    else:
        return f"([{_vector_to_str(pref_vector)}])"
