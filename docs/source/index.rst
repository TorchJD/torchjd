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
with multiple objectives.

.. important::
    This library is currently in an early development stage. The API is subject to significant changes
    in future versions. Use with caution in production environments and be prepared for potential
    breaking changes in upcoming releases.

API
===

- **backward**: Provides a function to compute the backward pass of an iteration of Jacobian descent.
- **aggregation**: Contains the implementation of aggregators such as UPGrad.

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    installation.md
    examples/index.rst

.. toctree::
    :maxdepth: 2
    :caption: API Reference
    :hidden:

    docs/main/backward.rst
    docs/main/aggregation/index.rst
