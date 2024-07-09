from torchjd.autojac._transform.base import Transform
from torchjd.autojac._transform.matrixify import Matrixify
from torchjd.autojac._transform.reshape import Reshape
from torchjd.autojac._transform.tensor_dict import (
    Gradients,
    GradientVectors,
    JacobianMatrices,
    Jacobians,
)


def make_aggregation(
    strategy: Transform[JacobianMatrices, GradientVectors]
) -> Transform[Jacobians, Gradients]:
    """TODO: doc"""

    return Reshape(strategy.required_keys) << strategy << Matrixify(strategy.required_keys)
