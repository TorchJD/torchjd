from typing import Iterable

from torch import Tensor

from torchjd._transform import Transform
from torchjd._transform._utils import ordered_set
from torchjd._transform.strategy._utils import _aggregate_group, _select_ordered_subdict
from torchjd._transform.tensor_dict import GradientVectors, JacobianMatrices
from torchjd.aggregation import Aggregator


class UnifyingStrategy(Transform[JacobianMatrices, GradientVectors]):
    """
    TODO: doc
    """

    def __init__(self, aggregator: Aggregator, key_order: Iterable[Tensor]):
        self.key_order = ordered_set(key_order)
        self.aggregator = aggregator

    def _compute(self, jacobian_matrices: JacobianMatrices) -> GradientVectors:
        """
        Concatenates the provided ``jacobian_matrices`` into a single matrix and aggregates it using
        the ``aggregator``. Returns the dictionary mapping each key from ``jacobian_matrices`` to
        the part of the obtained gradient vector, that corresponds to the jacobian matrix given for
        that key.

        :param jacobian_matrices: The dictionary of jacobian matrices to aggregate. The first
            dimension of each jacobian matrix should be the same.
        """
        ordered_matrices = _select_ordered_subdict(jacobian_matrices, self.key_order)
        return _aggregate_group(ordered_matrices, self.aggregator)

    def __str__(self) -> str:
        return f"Unifying {self.aggregator}"

    @property
    def required_keys(self) -> set[Tensor]:
        return set(self.key_order)

    @property
    def output_keys(self) -> set[Tensor]:
        return set(self.key_order)
