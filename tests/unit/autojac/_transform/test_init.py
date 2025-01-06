import torch

from torchjd.autojac._transform import EmptyTensorDict, Init

from ._dict_assertions import assert_tensor_dicts_are_close


def test_single_input():
    """
    Tests that when there is a single key to initialize, the Init transform creates a TensorDict
    whose value is a tensor full of ones, of the same shape as its key.
    """

    key = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input = EmptyTensorDict()

    init = Init([key])

    output = init(input)
    expected_output = {key: torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])}

    assert_tensor_dicts_are_close(output, expected_output)


def test_multiple_inputs():
    """
    Tests that when there are several keys to initialize, the Init transform creates a TensorDict
    whose values are tensors full of ones, of the same shape as their corresponding keys.
    """

    key1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    key2 = torch.tensor([1.0, 3.0, 5.0])
    input = EmptyTensorDict()

    init = Init([key1, key2])

    output = init(input)
    expected = {
        key1: torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        key2: torch.tensor([1.0, 1.0, 1.0]),
    }
    assert_tensor_dicts_are_close(output, expected)


def test_conjunction_of_inits_is_init():
    """
    Tests that the conjunction of 2 Init transforms is equivalent to a single Init transform with
    multiple keys.
    """

    x1 = torch.tensor(5.0)
    x2 = torch.tensor(6.0)
    input = EmptyTensorDict()

    init1 = Init([x1])
    init2 = Init([x2])
    conjunction_of_inits = init1 | init2
    init = Init([x1, x2])

    output = conjunction_of_inits(input)
    expected_output = init(input)

    assert_tensor_dicts_are_close(output, expected_output)
