from collections import OrderedDict

import pytest
import torch
from torch import Tensor

from torchjd.autojac._transform import Jacobians
from torchjd.autojac._transform.aggregation import UnifyingStrategy, _KeyType, _Matrixify, _Reshape
from torchjd.autojac._transform.tensor_dict import GradientVectors

from .utils import (
    EmptyDictProperty,
    ExpectedStructureProperty,
    aggregator,
    assert_tensor_dicts_are_close,
    keys,
)


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=keys)])
class TestUnifyingStructure(ExpectedStructureProperty):
    pass


@pytest.mark.parametrize("strategy", [UnifyingStrategy(aggregator, key_order=[])])
class TestUnifyingEmpty(EmptyDictProperty):
    pass


@pytest.mark.parametrize(
    ["united_gradient_vector", "jacobian_matrices"],
    [
        (
            torch.ones(10),
            {  # Total number of parameters according to the united gradient vector: 10
                torch.ones(5): torch.ones(2, 5),
                torch.ones(4): torch.ones(2, 3),
            },
        ),  # Total number of parameters according to the jacobian matrices: 9
        (
            torch.ones(10),
            {  # Total number of parameters according to the united gradient vector: 10
                torch.ones(5): torch.ones(2, 5),
                torch.ones(3): torch.ones(2, 3),
                torch.ones(3): torch.ones(2, 3),
            },
        ),  # Total number of parameters according to the jacobian matrices: 11
    ],
)
def test_disunite_wrong_vector_length(
    united_gradient_vector: Tensor, jacobian_matrices: dict[_KeyType, Tensor]
):
    with pytest.raises(ValueError):
        UnifyingStrategy._disunite(united_gradient_vector, OrderedDict(jacobian_matrices))


def test_matrixify():
    """Tests that the Matrixify transform correctly creates matrices from the jacobians."""

    n_outputs = 5
    key1 = torch.zeros([])
    key2 = torch.zeros([1])
    key3 = torch.zeros([2, 3])
    value1 = torch.tensor([1.0] * n_outputs)
    value2 = torch.tensor([[2.0]] * n_outputs)
    value3 = torch.tensor([[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]] * n_outputs)
    input = Jacobians({key1: value1, key2: value2, key3: value3})

    matrixify = _Matrixify([key1, key2, key3])

    output = matrixify(input)
    expected_output = {
        key1: torch.tensor([[1.0]] * n_outputs),
        key2: torch.tensor([[2.0]] * n_outputs),
        key3: torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]] * n_outputs),
    }

    assert_tensor_dicts_are_close(output, expected_output)


def test_reshape():
    """Tests that the Reshape transform correctly creates gradients from gradient vectors."""

    key1 = torch.zeros([])
    key2 = torch.zeros([1])
    key3 = torch.zeros([2, 3])
    value1 = torch.tensor([1.0])
    value2 = torch.tensor([2.0])
    value3 = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    input = GradientVectors({key1: value1, key2: value2, key3: value3})

    reshape = _Reshape([key1, key2, key3])

    output = reshape(input)
    expected_output = {
        key1: torch.tensor(1.0),
        key2: torch.tensor([2.0]),
        key3: torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
    }

    assert_tensor_dicts_are_close(output, expected_output)
