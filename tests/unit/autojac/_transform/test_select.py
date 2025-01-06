import torch

from torchjd.autojac._transform import Select, TensorDict

from ._dict_assertions import assert_tensor_dicts_are_close


def test_partition():
    """
    Tests that the Select transform works correctly by applying 2 different Selects to a TensorDict,
    whose keys form a partition of the keys of the TensorDict.
    """

    key1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    key2 = torch.tensor([1.0, 3.0, 5.0])
    key3 = torch.tensor(2.0)
    value1 = torch.ones_like(key1)
    value2 = torch.ones_like(key2)
    value3 = torch.ones_like(key3)
    input = TensorDict({key1: value1, key2: value2, key3: value3})

    select1 = Select([key1, key2], [key1, key2, key3])
    select2 = Select([key3], [key1, key2, key3])

    output1 = select1(input)
    expected_output1 = {key1: value1, key2: value2}

    assert_tensor_dicts_are_close(output1, expected_output1)

    output2 = select2(input)
    expected_output2 = {key3: value3}

    assert_tensor_dicts_are_close(output2, expected_output2)


def test_conjunction_of_selects_is_select():
    """
    Tests that the conjunction of 2 Select transforms is equivalent to directly using a Select with
    the union of the keys of the 2 Selects.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    x3 = torch.tensor(7.0)
    input = TensorDict({x1: torch.ones_like(x1), x2: torch.ones_like(x2), x3: torch.ones_like(x3)})

    select1 = Select([x1], [x1, x2, x3])
    select2 = Select([x2], [x1, x2, x3])
    conjunction_of_selects = select1 | select2
    select = Select([x1, x2], [x1, x2, x3])

    output = conjunction_of_selects(input)
    expected_output = select(input)

    assert_tensor_dicts_are_close(output, expected_output)
