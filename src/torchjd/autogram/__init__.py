"""
The autogram package enables the activation of Gramian-based Jacobian descent on your models (As
described in Section 6 of `Jacobian Descent For Multi-Objective Optimization
<https://arxiv.org/pdf/2406.16232>`_). It provides a convenient way to modify a model's backward
pass, allowing you to seamlessly integrate multi-objective optimization in your PyTorch code.

This method typically provides a memory improvement over the :doc:`autojac <../autojac/index>`
package which typically leads to time improvement.

# TODO improve:
The exhaustive list of supported Weightings compatible with `autogram` is:

* :class:`~torchjd.aggregation.UPGradWeighting`
* :class:`~torchjd.aggregation.AlignedMTLWeighting`
* :class:`~torchjd.aggregation.CAGradWeighting`
* :class:`~torchjd.aggregation.ConstantWeighting`
* :class:`~torchjd.aggregation.DualProjWeighting`
* :class:`~torchjd.aggregation.IMTLGWeighting`
* :class:`~torchjd.aggregation.KrumWeighting`
* :class:`~torchjd.aggregation.MeanWeighting`
* :class:`~torchjd.aggregation.MGDAWeighting`
* :class:`~torchjd.aggregation.PCGradWeighting`
* :class:`~torchjd.aggregation.RandomWeighting`
* :class:`~torchjd.aggregation.SumWeighting`
"""

from ._gramian_reverse_accumulator import GramianReverseAccumulator
