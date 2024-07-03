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

TorchJD is a library enabling Jacobian descent with PyTorch, to train neural networks with multiple
objectives. It is based on the theory from `Jacobian Descent For Multi-Objective Optimization
<https://arxiv.org/pdf/2406.16232>`_ and several other related publications. The main purpose is to
jointly optimize multiple objectives without combining them into a single scalar loss. Using TorchJD
to minimize a vector of objectives comes at the expense of some computational overhead compared to
directly using PyTorch to minimize a scalarization of the objectives. However when the objectives
are conflicting, this can be essential. To get started, check out our
:doc:`basic usage example <examples/basic_usage>`.


.. important::
    This library is currently in an early development stage. The API is subject to significant
    changes in future versions.


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

    docs/autojac/backward.rst
    docs/autojac/mtl_backward.rst
    docs/aggregation/index.rst
