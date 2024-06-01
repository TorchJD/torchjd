import torch
from unit.transform.utils import assert_tensor_dicts_are_close

from torchjd.transform import Subset, TensorDict


def test_subset_partition():
    """
    Tests that the Subset transform works correctly by applying 2 different Subsets to a
    TensorDict, whose keys form a partition of the keys of the TensorDict.
    """

    key1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    key2 = torch.tensor([1.0, 3.0, 5.0])
    key3 = torch.tensor(2.0)
    value1 = torch.ones_like(key1)
    value2 = torch.ones_like(key2)
    value3 = torch.ones_like(key3)
    input = TensorDict({key1: value1, key2: value2, key3: value3})

    subset1 = Subset([key1, key2], [key1, key2, key3])
    subset2 = Subset([key3], [key1, key2, key3])

    output1 = subset1(input)
    expected_output1 = {key1: value1, key2: value2}

    assert_tensor_dicts_are_close(output1, expected_output1)

    output2 = subset2(input)
    expected_output2 = {key3: value3}

    assert_tensor_dicts_are_close(output2, expected_output2)


def test_conjunction_of_subsets_is_subset():
    """
    Tests that the conjunction of 2 Subset transforms is equivalent to directly using a Subset with
    the union of the keys of the 2 Subsets.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    x3 = torch.tensor(7.0)
    input = TensorDict({x1: torch.ones_like(x1), x2: torch.ones_like(x2), x3: torch.ones_like(x3)})

    subset1 = Subset([x1], [x1, x2, x3])
    subset2 = Subset([x2], [x1, x2, x3])
    conjunction_of_subsets = subset1 | subset2
    subset = Subset([x1, x2], [x1, x2, x3])

    output = conjunction_of_subsets(input)
    expected_output = subset(input)

    assert_tensor_dicts_are_close(output, expected_output)
