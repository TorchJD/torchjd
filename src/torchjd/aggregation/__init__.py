r"""
A mapping :math:`\mathcal A: \mathbb R^{m\times n} \to \mathbb R^n` reducing any matrix
:math:`J \in \mathbb R^{m\times n}` into its aggregation :math:`\mathcal A(J) \in \mathbb R^n` is
called an aggregator.

In the context of JD, the matrix to aggregate is a Jacobian whose rows are the gradients of the
individual objectives. The aggregator is used to reduce this matrix into an update vector for the
parameters of the model

In TorchJD, an aggregator is a class that inherits from the abstract class
:class:`~torchjd.aggregation._aggregator_bases.Aggregator`. We provide the following list of
aggregators from the literature:

.. role:: raw-html(raw)
   :format: html

.. |yes| replace:: :raw-html:`<center><font color="#28b528">✔</font></center>`
.. |no| replace:: :raw-html:`<center><font color="#e63232">✘</font></center>`

.. list-table::
   :widths: 25 15 15 15
   :header-rows: 1

   * - :doc:`Aggregator <index>`
     - :ref:`Non-conflicting <Non-conflicting>`
     - :ref:`Linear under scaling <Linear under scaling>`
     - :ref:`Weighted <Weighted>`
   * - :doc:`UPGrad <upgrad>` (recommended)
     - |yes|
     - |yes|
     - |yes|
   * - :doc:`Aligned-MTL <aligned_mtl>`
     - |no|
     - |no|
     - |yes|
   * - :doc:`CAGrad <cagrad>`
     - |no|
     - |no|
     - |yes|
   * - :doc:`ConFIG <config>`
     - |no|
     - |yes|
     - |yes|
   * - :doc:`Constant <constant>`
     - |no|
     - |yes|
     - |yes|
   * - :doc:`DualProj <dualproj>`
     - |yes|
     - |no|
     - |yes|
   * - :doc:`GradDrop <graddrop>`
     - |no|
     - |no|
     - |no|
   * - :doc:`IMTL-G <imtl_g>`
     - |no|
     - |no|
     - |yes|
   * - :doc:`Krum <krum>`
     - |no|
     - |no|
     - |yes|
   * - :doc:`Mean <mean>`
     - |no|
     - |yes|
     - |yes|
   * - :doc:`MGDA <mgda>`
     - |yes|
     - |no|
     - |yes|
   * - :doc:`Nash-MTL <nash_mtl>`
     - |yes|
     - |no|
     - |yes|
   * - :doc:`PCGrad <pcgrad>`
     - |no|
     - |yes|
     - |yes|
   * - :doc:`Random <random>`
     - |no|
     - |yes|
     - |yes|
   * - :doc:`Sum <sum>`
     - |no|
     - |yes|
     - |yes|
   * - :doc:`Trimmed Mean <trimmed_mean>`
     - |no|
     - |no|
     - |no|

.. hint::
    This table is an adaptation of the one available in `Jacobian Descent For Multi-Objective
    Optimization <https://arxiv.org/pdf/2406.16232>`_. The paper provides precise justification of
    the properties in Section 2.2 as well as proofs in Appendix B.

.. _Non-conflicting:
.. admonition::
    Non-conflicting

    An aggregator :math:`\mathcal A: \mathbb R^{m\times n} \to \mathbb R^n` is said to be
    *non-conflicting* if for any :math:`J\in\mathbb R^{m\times n}`, :math:`J\cdot\mathcal A(J)` is a
    vector with only non-negative elements.

    In other words, :math:`\mathcal A` is non-conflicting whenever the aggregation of any matrix has
    non-negative inner product with all rows of that matrix. In the context of JD, this ensures that
    no objective locally increases.

.. _Linear under scaling:
.. admonition::
    Linear under scaling

    An aggregator :math:`\mathcal A: \mathbb R^{m\times n} \to \mathbb R^n` is said to be
    *linear under scaling* if for any :math:`J\in\mathbb R^{m\times n}`, the mapping from any
    positive :math:`c\in\mathbb R^{n}` to :math:`\mathcal A(\operatorname{diag}(c)\cdot J)` is
    linear in :math:`c`.

    In other words, :math:`\mathcal A` is linear under scaling whenever scaling a row of the matrix
    to aggregate scales its influence proportionally. In the context of JD, this ensures that even
    when the gradient norms are imbalanced, each gradient will contribute to the update
    proportionally to its norm.

.. _Weighted:
.. admonition::
    Weighted

    An aggregator :math:`\mathcal A: \mathbb R^{m\times n} \to \mathbb R^n` is said to be *weighted*
    if for any :math:`J\in\mathbb R^{m\times n}`, there exists a weight vector
    :math:`w\in\mathbb R^m` such that :math:`\mathcal A(J)=J^\top w`.

    In other words, :math:`\mathcal A` is weighted whenever the aggregation of any matrix is always
    in the span of the rows of that matrix. This ensures a higher precision of the Taylor
    approximation that JD relies on.
"""

from ._aggregator_bases import Aggregator, GramianWeightedAggregator, WeightedAggregator
from ._aligned_mtl import AlignedMTL, AlignedMTLWeighting
from ._config import ConFIG
from ._constant import Constant, ConstantWeighting
from ._dualproj import DualProj, DualProjWeighting
from ._graddrop import GradDrop
from ._imtl_g import IMTLG, IMTLGWeighting
from ._krum import Krum, KrumWeighting
from ._mean import Mean, MeanWeighting
from ._mgda import MGDA, MGDAWeighting
from ._pcgrad import PCGrad, PCGradWeighting
from ._random import Random, RandomWeighting
from ._sum import Sum, SumWeighting
from ._trimmed_mean import TrimmedMean
from ._upgrad import UPGrad, UPGradWeighting
from ._utils.check_dependencies import (
    OptionalDepsNotInstalledError as _OptionalDepsNotInstalledError,
)
from ._weighting_bases import Matrix, PSDMatrix, Weighting

try:
    from ._cagrad import CAGrad, CAGradWeighting
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass

try:
    from ._nash_mtl import NashMTL
except _OptionalDepsNotInstalledError:  # The required dependencies are not installed
    pass
