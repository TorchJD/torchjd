from conflictfree.grad_operator import ConFIGOperator
from conflictfree.length_model import LengthModel, ProjectionLength
from conflictfree.weight_model import EqualWeight, WeightModel
from torch import Tensor

from torchjd.aggregation.bases import Aggregator


class ConFIG(Aggregator):
    """
    :class:`~torchjd.aggregation.bases.Aggregator` as defined in Equation 2 of `ConFIG: Towards
    Conflict-free Training of Physics Informed Neural Networks <https://arxiv.org/pdf/2408.11104>`_.

    .. admonition::
        Example

        Use ConFIG to aggregate a matrix.

        >>> from torch import tensor
        >>> from torchjd.aggregation import ConFIG
        >>>
        >>> A = ConFIG()
        >>> J = tensor([[-4., 1., 1.], [6., 1., 1.]])
        >>>
        >>> A(J)
        # TODO: add result
        # TODO: add doc test
    """

    def __init__(
        self,
        weight_model: WeightModel = EqualWeight(),
        length_model: LengthModel = ProjectionLength(),
        allow_simplified_model: bool = True,
        use_least_square: bool = True,
    ):
        super().__init__()
        self._config_operator = ConFIGOperator(
            weight_model=weight_model,
            length_model=length_model,
            allow_simplified_model=allow_simplified_model,
            use_least_square=use_least_square,
        )

    def forward(self, matrix: Tensor) -> Tensor:
        return self._config_operator.calculate_gradient(matrix)
