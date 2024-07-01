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
- :doc:`Multi-task Learning (MTL) <multi_task>` provides an example of multi-task learning where the
  aggregation is performed only on the shared parameters. This is performed by using the dedicated
  backpropagation function :doc:`torchjd.multi_task_backward <../docs/autojac/multi_task_backward>`.

.. toctree::
    :hidden:

    basic_usage.rst
    iwrm.rst
    multi_task.rst
