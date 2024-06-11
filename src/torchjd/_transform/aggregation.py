from torchjd._transform.base import Transform
from torchjd._transform.matrixify import Matrixify
from torchjd._transform.reshape import Reshape
from torchjd._transform.tensor_dict import Gradients, GradientVectors, JacobianMatrices, Jacobians


def make_aggregation(
    strategy: Transform[JacobianMatrices, GradientVectors]
) -> Transform[Jacobians, Gradients]:
    """TODO: doc"""

    return Reshape(strategy.required_keys) << strategy << Matrixify(strategy.required_keys)
