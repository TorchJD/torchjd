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
<https://arxiv.org/pdf/2406.16232>`_ and several other related publications.

The main purpose is to jointly optimize multiple objectives without combining them into a single
scalar loss. When the objectives are conflicting, this can be the key to a successful and stable
optimization. To get started, check out our :doc:`basic usage example
<examples/basic_usage>`.

Gradient descent relies on gradients to optimize a single objective. Jacobian descent takes this
idea a step further, using the Jacobian to optimize multiple objectives. An important component of
Jacobian descent is the aggregator, which maps the Jacobian to an optimization step. In the page
:doc:`Aggregation <docs/aggregation/index>`, we provide an overview of the various aggregators available
in TorchJD, along with some of their key characteristics. A precise description of this formalism
along with the :doc:`UPGrad <docs/aggregation/upgrad>` aggregator is available in Section 2 of `Jacobian
Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

A straightforward application of Jacobian descent is multi-task learning, in which the vector of
per-task losses has to be minimized. To start using TorchJD for multi-task learning, follow our
:doc:`MTL example <examples/mtl>`.

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
