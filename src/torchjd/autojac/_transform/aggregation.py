from .base import Transform
from .matrixify import Matrixify
from .reshape import Reshape
from .tensor_dict import Gradients, GradientVectors, JacobianMatrices, Jacobians


def make_aggregation(
    strategy: Transform[JacobianMatrices, GradientVectors]
) -> Transform[Jacobians, Gradients]:
    """TODO: doc"""

    return Reshape(strategy.required_keys) << strategy << Matrixify(strategy.required_keys)
