:hide-toc:

|

.. image:: _static/logo-dark-mode.png
    :width: 400
    :alt: torchjd
    :align: center
    :class: only-dark, no-scaled-link

.. image:: _static/logo-light-mode.png
    :width: 400
    :alt: torchjd
    :align: center
    :class: only-light, no-scaled-link

|

TorchJD is a library enabling Jacobian descent with PyTorch, for optimization of neural networks
with multiple objectives. It is based on the theory from `Jacobian Descent For Multi-Objective
Optimization <https://arxiv.org/pdf/2406.16232>`_ and it contains algorithms from many other related
papers.

.. important::
    This library is currently in an early development stage. The API is subject to significant changes
    in future versions. Use with caution in production environments and be prepared for potential
    breaking changes in upcoming releases.

TorchJD is open-source, under MIT License. The source code is available on
`GitHub <https://github.com/TorchJD/torchjd>`_.

.. toctree::
    :caption: Getting Started
    :hidden:

    installation.md
    examples/index.rst

.. toctree::
    :caption: API Reference
    :hidden:

    docs/backward.rst
    docs/aggregation/index.rst
