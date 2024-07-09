Examples
========

This section contains some usage examples for TorchJD.

- :doc:`Basic Usage <basic_usage>` provides a toy example using :doc:`torchjd.backward
  <../docs/autojac/backward>` to make a step of Jacobian descent with the :doc:`UPGrad
  <../docs/aggregation/upgrad>` aggregator.
- :doc:`Instance-Wise Risk Minimization (IWRM) <iwrm>` provides an example in which we minimize the
  vector of per-instance losses, using stochastic sub-Jacobian descent (SSJD). It is compared to the
  usual minimization of the average loss, called empirical risk minimization (ERM), using stochastic
  gradient descent (SGD).
- :doc:`Multi-Task Learning (MTL) <mtl>` provides an example of multi-task learning where Jacobian
  descent is used to optimize the vector of per-task losses of a multi-task model, using the
  dedicated backpropagation function :doc:`mtl_backward <../docs/autojac/mtl_backward>`.

.. toctree::
    :hidden:

    basic_usage.rst
    iwrm.rst
    mtl.rst
