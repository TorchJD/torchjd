from torchjd.transform.base import Transform
from torchjd.transform.matrixify import Matrixify
from torchjd.transform.reshape import Reshape
from torchjd.transform.tensor_dict import Gradients, GradientVectors, JacobianMatrices, Jacobians


def make_aggregation(
    strategy: Transform[JacobianMatrices, GradientVectors]
) -> Transform[Jacobians, Gradients]:
    """TODO: doc"""

    return Reshape(strategy.required_keys) << strategy << Matrixify(strategy.required_keys)
