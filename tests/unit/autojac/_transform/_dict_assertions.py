from torch import Tensor
from torch.testing import assert_close

from torchjd._autojac._transform import TensorDict


def assert_tensor_dicts_are_close(d1: dict[Tensor, Tensor], d2: dict[Tensor, Tensor]) -> None:
    """
    Check that two dictionaries of tensors are close enough. Note that this does not require the
    keys to have the same ordering.

    Additionally, if the parameters are TensorDicts, this checks that their content respects their
    supposed type.
    """

    if isinstance(d1, TensorDict):
        d1.check()

    if isinstance(d2, TensorDict):
        d2.check()

    assert d1.keys() == d2.keys()

    for key in d1:
        assert_close(d1[key], d2[key])
