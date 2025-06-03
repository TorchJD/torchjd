from torch.testing import assert_close

from torchjd._autojac._transform._base import TensorDict


def assert_tensor_dicts_are_close(d1: TensorDict, d2: TensorDict) -> None:
    """
    Check that two dictionaries of tensors are close enough. Note that this does not require the
    keys to have the same ordering.
    """

    assert d1.keys() == d2.keys()

    for key in d1:
        assert_close(d1[key], d2[key])
