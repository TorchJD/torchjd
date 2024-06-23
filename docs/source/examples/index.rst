Examples
========

This section contains some usage examples for TorchJD.

- :doc:`Basic Usage <basic_usage>` provides a toy example using :doc:`torchjd.backward
  <../docs/backward>` to make a step of Jacobian descent with the :doc:`UPGrad
  <../docs/aggregation/upgrad>` aggregator.
- :doc:`Instance-Wise Risk Minimization (IWRM) <iwrm>` provides an example in which we minimize the
  vector of per-instance losses, using stochastic sub-Jacobian descent (SSJD). It is compared to the
  usual minimization of the average loss, called empirical risk minimization (ERM), using stochastic
  gradient descent (SGD).

.. toctree::
    :hidden:

    basic_usage.rst
    iwrm.rst
