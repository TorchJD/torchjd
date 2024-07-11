from torch import Tensor

from .constant import _ConstantWeighting
from .mean import _MeanWeighting


def _check_pref_vector(pref_vector: Tensor | None) -> None:
    """Checks the correctness of the parameter pref_vector."""

    if pref_vector is not None:
        if pref_vector.ndim != 1:
            raise ValueError(
                "Parameter `pref_vector` must be a vector (1D Tensor). Found `pref_vector.ndim = "
                f"{pref_vector.ndim}`."
            )


def _pref_vector_to_weighting(pref_vector: Tensor | None) -> _ConstantWeighting | _MeanWeighting:
    """Returns the weighting associated to a given preference vector."""

    if pref_vector is None:
        weighting = _MeanWeighting()
    else:
        weighting = _ConstantWeighting(pref_vector)

    return weighting
