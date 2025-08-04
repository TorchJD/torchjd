"""
This package enable Jacobian descent, through the `backward` and `mtl_backward` functions, which
are meant to replace the call to `torch.backward` or `loss.backward` in gradient descent. To combine
the information of the Jacobian, an aggregator from the `aggregation` package has to be used.
"""

from ._autogram import HandleManager, augment_model_with_iwrm_autogram
from ._autojac import backward, mtl_backward
