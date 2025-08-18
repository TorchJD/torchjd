"""
The autogram package enables the activation of Gramian-based Jacobian descent on your models (As
described in Section 6 of `Jacobian Descent For Multi-Objective Optimization
<https://arxiv.org/pdf/2406.16232>`_). It provides a convenient way to modify a model's backward
pass, allowing you to seamlessly integrate multi-objective optimization in your PyTorch code.

This method typically provides a memory improvement over the :doc:`autojac <../autojac/index>`
package which typically leads to time improvement.
"""

from ._augment_model import augment_model_for_iwrm
from ._handle import RemovableHandle
