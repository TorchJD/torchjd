import torch
from pytest import mark
from torch import Tensor
from torch.testing import assert_close

from torchjd.aggregation import Aggregator

from ._inputs import matrices, scaled_matrices, stationary_matrices, zero_rank_matrices


class ExpectedStructureProperty:
    """
    This class tests that the vector returned by the `__call__` method of an `Aggregator` has the
    expected structure: it should return a vector whose dimension should be the number of columns of
    the input matrix, and that should only contain finite values (no `nan`, `inf` or `-inf`). Note
    that this property implies that the `__call__` method does not raise any exception.
    """

    @classmethod
    @mark.parametrize("matrix", scaled_matrices + zero_rank_matrices)
    def test_expected_structure_property(cls, aggregator: Aggregator, matrix: Tensor):
        cls._assert_expected_structure_property(aggregator, matrix)

    @staticmethod
    def _assert_expected_structure_property(aggregator: Aggregator, matrix: Tensor) -> None:
        vector = aggregator(matrix)  # Will fail if the call raises an exception
        assert vector.shape == matrix.shape[1:]
        assert vector.isfinite().all()


class NonConflictingProperty:
    """
    This class tests empirically that a given `Aggregator` satisfies the non-conflicting property.
    """

    @classmethod
    @mark.parametrize("matrix", stationary_matrices + matrices)
    def test_non_conflicting_property(
        cls,
        aggregator: Aggregator,
        matrix: Tensor,
    ):
        cls._assert_non_conflicting_property(aggregator, matrix)

    @staticmethod
    def _assert_non_conflicting_property(
        aggregator: Aggregator,
        matrix: Tensor,
    ) -> None:
        vector = aggregator(matrix)
        output_direction = matrix @ vector
        positive_directions = output_direction[output_direction >= 0]
        assert_close(positive_directions.norm(), output_direction.norm(), atol=4e-04, rtol=0)


class PermutationInvarianceProperty:
    """
    This class tests empirically that for a given `Aggregator`, randomly permuting rows of the input
    matrix doesn't change the aggregation.
    """

    N_PERMUTATIONS = 5

    @classmethod
    @mark.parametrize("matrix", matrices)
    def test_permutation_invariance_property(cls, aggregator: Aggregator, matrix: Tensor):
        cls._assert_permutation_invariance_property(aggregator, matrix)

    @staticmethod
    def _assert_permutation_invariance_property(aggregator: Aggregator, matrix: Tensor) -> None:
        vector = aggregator(matrix)

        for _ in range(PermutationInvarianceProperty.N_PERMUTATIONS):
            permuted_matrix = PermutationInvarianceProperty._permute_randomly(matrix)
            permuted_vector = aggregator(permuted_matrix)

            assert_close(vector, permuted_vector, atol=5e-04, rtol=1e-05)

    @staticmethod
    def _permute_randomly(matrix: Tensor) -> Tensor:
        row_permutation = torch.randperm(matrix.size(dim=0))
        return matrix[row_permutation]
