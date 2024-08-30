Aggregation
===========

A mapping :math:`\mathcal A: \mathbb R^{m\times n} \to \mathbb R^n` reducing any matrix
:math:`J \in \mathbb R^{m\times n}` into its aggregation :math:`\mathcal A(J) \in \mathbb R^n` is
called an aggregator. In the context of Jacobian descent, an aggregator is typically used to reduce
the Jacobian of the objectives into an update vector for the parameters of the model.

This package provides several aggregators from the literature:

.. list-table:: Properties of aggregators
   :widths: 25 25 25 25
   :header-rows: 1

   * - :doc:`Aggregator (abstract) <bases>`
     - Non-conflicting
     - Linear under scaling
     - Weighted
   * - :doc:`Aligned-MTL <aligned_mtl>`
     - ✘
     - ✘
     - ✔
   * - :doc:`CAGrad <cagrad>`
     - ✘
     - ✘
     - ✔
   * - :doc:`Constant <constant>`
     - ✘
     - ✔
     - ✔
   * - :doc:`DualProj <dualproj>`
     - ✔
     - ✘
     - ✔
   * - :doc:`GradDrop <graddrop>`
     - ✘
     - ✘
     - ✘
   * - :doc:`IMTL-G <imtl_g>`
     - ✘
     - ✘
     - ✔
   * - :doc:`Krum <krum>`
     - ?
     - ?
     - ?
   * - :doc:`Mean <mean>`
     - ✘
     - ✔
     - ✔
   * - :doc:`MGDA <mgda>`
     - ✔
     - ✘
     - ✔
   * - :doc:`Nash-MTL <nash_mtl>`
     - ✔
     - ✘
     - ✔
   * - :doc:`PCGrad <pcgrad>`
     - ✘
     - ✔
     - ✔
   * - :doc:`Random <random>`
     - ✘
     - ✔
     - ✔
   * - :doc:`Sum <sum>`
     - ✘
     - ✔
     - ✔
   * - :doc:`Trimmed Mean <trimmed_mean>`
     - ?
     - ?
     - ?
   * - :doc:`UPGrad <upgrad>`
     - ✔
     - ✔
     - ✔

.. hint::
    This table is an adaptation of the one available in `Jacobian Descent For Multi-Objective
    Optimization <https://arxiv.org/pdf/2406.16232>`_. The paper provides the proofs in Appendix B.


.. toctree::
    :hidden:
    :maxdepth: 1

    bases.rst
    aligned_mtl.rst
    cagrad.rst
    constant.rst
    dualproj.rst
    graddrop.rst
    imtl_g.rst
    krum.rst
    mean.rst
    mgda.rst
    nash_mtl.rst
    pcgrad.rst
    random.rst
    sum.rst
    trimmed_mean.rst
    upgrad.rst
