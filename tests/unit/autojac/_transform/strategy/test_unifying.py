from collections import OrderedDict

import pytest
import torch
from torch import Tensor

from torchjd.autojac._transform.strategy import UnifyingStrategy
from torchjd.autojac._transform.strategy.unifying import _disunite, _KeyType

from .utils import EmptyDictProperty, ExpectedStructureProperty, aggregator, keys


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
def test__disunite_wrong_vector_length(
    united_gradient_vector: Tensor, jacobian_matrices: dict[_KeyType, Tensor]
):
    with pytest.raises(ValueError):
        _disunite(united_gradient_vector, OrderedDict(jacobian_matrices))
