Aggregation
===========

A mapping :math:`\mathcal A: \mathbb R^{m\times n} \to \mathbb R^n` reducing any matrix
:math:`J \in \mathbb R^{m\times n}` into its aggregation :math:`\mathcal A(J) \in \mathbb R^n` is
called an aggregator.In the context of Jacobian descent, an aggregator is typically used to reduce
the Jacobian of the objectives into an update vector for the parameters of the model.

This package provides several aggregators from the literature:

.. toctree::
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
