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
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`CAGrad <cagrad>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`Constant <constant>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`DualProj <dualproj>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`GradDrop <graddrop>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
   * - :doc:`IMTL-G <imtl_g>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`Krum <krum>`
     - :raw-html:`<center><font color="yellow">?</font></center>`
     - :raw-html:`<center><font color="yellow">?</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`Mean <mean>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`MGDA <mgda>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`Nash-MTL <nash_mtl>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`PCGrad <pcgrad>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`Random <random>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`Sum <sum>`
     - :raw-html:`<center><font color="red">✘</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
   * - :doc:`Trimmed Mean <trimmed_mean>`
     - :raw-html:`<center><font color="yellow">?</font></center>`
     - :raw-html:`<center><font color="yellow">?</font></center>`
     - :raw-html:`<center><font color="red">✘</font></center>`
   * - :doc:`UPGrad <upgrad>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`
     - :raw-html:`<center><font color="green">✔</font></center>`

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
