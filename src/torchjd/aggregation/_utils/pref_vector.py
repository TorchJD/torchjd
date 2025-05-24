from torch import Tensor

from torchjd.aggregation._weighting_bases import Matrix, Weighting
from torchjd.aggregation.constant import _ConstantWeighting

from .str import vector_to_str


def pref_vector_to_weighting(
    pref_vector: Tensor | None, default: Weighting[Matrix]
) -> Weighting[Matrix]:
    """
    Returns the weighting associated to a given preference vector, with a fallback to a default
    weighting if the preference vector is None.
    """

    if pref_vector is None:
        return default
    else:
        if pref_vector.ndim != 1:
            raise ValueError(
                "Parameter `pref_vector` must be a vector (1D Tensor). Found `pref_vector.ndim = "
                f"{pref_vector.ndim}`."
            )
        return _ConstantWeighting(pref_vector)


def pref_vector_to_str_suffix(pref_vector: Tensor | None) -> str:
    """Returns a suffix string containing the representation of the optional preference vector."""

    if pref_vector is None:
        return ""
    else:
        return f"([{vector_to_str(pref_vector)}])"
