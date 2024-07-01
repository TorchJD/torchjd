import pytest
from unit.autojac._transform.strategy.utils import jacobian_matrix_dicts

from torchjd.autojac._transform import Transform
from torchjd.autojac._transform.tensor_dict import GradientVectors, JacobianMatrices


class ExpectedStructureProperty:
    """
    This class tests that the `__call__` method of a strategy returns a dictionary of gradients with
    the expected structure:
    - It has the same set of keys as the input dictionary of jacobians.
    - The shape of each of its gradients is equal to the shape of the corresponding jacobian, minus
      the first dimension.
    """

    @classmethod
    @pytest.mark.parametrize("jacobian_matrices", jacobian_matrix_dicts)
    def test_expected_structure_property(
        cls,
        strategy: Transform[JacobianMatrices, GradientVectors],
        jacobian_matrices: JacobianMatrices,
    ):
        cls._assert_expected_structure(strategy, jacobian_matrices)

    @staticmethod
    def _assert_expected_structure(
        strategy: Transform[JacobianMatrices, GradientVectors],
        jacobian_matrices: JacobianMatrices,
    ) -> None:
        gradient_vectors = strategy(jacobian_matrices)
        ExpectedStructureProperty._assert_expected_keys(jacobian_matrices, gradient_vectors)
        ExpectedStructureProperty._assert_expected_shapes(jacobian_matrices, gradient_vectors)

    @staticmethod
    def _assert_expected_keys(
        jacobian_matrices: JacobianMatrices, gradient_vectors: GradientVectors
    ):
        assert set(jacobian_matrices.keys()) == set(gradient_vectors.keys())

    @staticmethod
    def _assert_expected_shapes(
        jacobian_matrices: JacobianMatrices, gradient_vectors: GradientVectors
    ):
        for key in jacobian_matrices.keys():
            jacobian_matrix = jacobian_matrices[key]
            gradient_vector = gradient_vectors[key]
            assert gradient_vector.numel() == jacobian_matrix[0].numel()


class EmptyDictProperty:
    """
    This class tests that the `__call__` method of a strategy applied to an empty dict, returns an
    empty dict.
    """

    @classmethod
    def test_empty_dict_property(
        cls,
        strategy: Transform[JacobianMatrices, GradientVectors],
    ):
        cls._assert_empty_dict(strategy)

    @staticmethod
    def _assert_empty_dict(
        strategy: Transform[JacobianMatrices, GradientVectors],
    ) -> None:
        gradient_vectors = strategy(JacobianMatrices({}))
        assert len(gradient_vectors) == 0
