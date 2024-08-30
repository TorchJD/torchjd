Aggregation
===========

A mapping :math:`\mathcal A: \mathbb R^{m\times n} \to \mathbb R^n` reducing any matrix
:math:`J \in \mathbb R^{m\times n}` into its aggregation :math:`\mathcal A(J) \in \mathbb R^n` is
called an aggregator. In the context of Jacobian descent, an aggregator is typically used to reduce
the Jacobian of the objectives into an update vector for the parameters of the model.

This package provides several aggregators from the literature:

.. role:: raw-html(raw)
   :format: html

.. list-table:: Properties of aggregators
   :widths: 25 15 15 15
   :header-rows: 1

   * - :doc:`Aggregator (abstract) <bases>`
     - Non-conflicting
     - Linear under scaling
     - Weighted
   * - :doc:`Aligned-MTL <aligned_mtl>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`CAGrad <cagrad>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`Constant <constant>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`DualProj <dualproj>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`GradDrop <graddrop>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="red">✘</font>`
   * - :doc:`IMTL-G <imtl_g>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`Krum <krum>`
     - ?
     - ?
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`Mean <mean>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`MGDA <mgda>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`Nash-MTL <nash_mtl>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`PCGrad <pcgrad>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`Random <random>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`Sum <sum>`
     - :raw-html:`<font color="red">✘</font>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="green">✔</font>`
   * - :doc:`Trimmed Mean <trimmed_mean>`
     - ?
     - ?
     - :raw-html:`<font color="red">✘</font>`
   * - :doc:`UPGrad <upgrad>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="green">✔</font>`
     - :raw-html:`<font color="green">✔</font>`

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
