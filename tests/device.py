import os

import torch

_POSSIBLE_TEST_DEVICES = {"cpu", "cuda:0", "mps"}

try:
    _device_str = os.environ["PYTEST_TORCH_DEVICE"]
except KeyError:
    _device_str = "cpu"  # Default to cpu if environment variable not set

if _device_str not in _POSSIBLE_TEST_DEVICES:
    raise ValueError(
        f"Invalid value of environment variable PYTEST_TORCH_DEVICE: {_device_str}.\n"
        f"Possible devices: {_POSSIBLE_TEST_DEVICES}"
    )

if _device_str == "cuda:0" and not torch.cuda.is_available():
    raise ValueError('Requested device "cuda:0" but cuda is not available.')

if _device_str == "mps":
    # Check that MPS is available (following https://docs.pytorch.org/docs/stable/notes/mps.html)
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            raise ValueError(
                "MPS not available because the current PyTorch install was not built with MPS "
                "enabled."
            )
        else:
            raise ValueError(
                "MPS not available because the current MacOS version is not 12.3+ and/or you do not"
                " have an MPS-enabled device on this machine."
            )

DEVICE = torch.device(_device_str)
