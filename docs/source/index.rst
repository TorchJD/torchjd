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
    :name: api reference
    :caption: API Reference
    :hidden:

    docs/main/backward.rst
    docs/main/aggregation/index.rst

Usage
=====

This example shows how to use torchjd to perform an iteration of Jacobian Descent on a regression
model. In this example, a batch of inputs is forwarded through the model and the corresponding batch
of labels is used to compute a batch of losses. These losses are then backwarded through the model.
The obtained Jacobian matrix, consisting of the gradients of the losses, is then aggregated using
UPGrad, and the parameters are updated using the resulting aggregation.

Import several classes from torch and torchjd:

>>> import torch
>>> from torch.nn import MSELoss, Sequential, Linear, ReLU
>>> from torch.optim import SGD
>>>
>>> import torchjd
>>> from torchjd.aggregation import UPGrad

Define the model and the optimizer, as usual:

>>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
>>> optimizer = SGD(model.parameters(), lr=0.1)

Define the aggregator that will be used to combine the Jacobian matrix:

>>> A = UPGrad()

In essence, UPGrad projects each gradient onto the dual cone of the rows of the Jacobian and
averages the results. This ensures that locally, no loss will be negatively affected by the update.

Now that everything is defined, we can train the model. Define the input and the associated target:

>>> input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
>>> target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

Here, we generate fake data in which each target is equal to the sum of its corresponding input
vector, for the sake of the example.

We can now compute the losses associated to each element of the batch.

>>> loss_fn = MSELoss(reduction='none')
>>> output = model(input)
>>> losses = loss_fn(output, target)

Note that setting `reduction='none'` is necessary to obtain the element-wise loss vector.

The last steps are similar to gradient descent-based optimization.

Reset the ``.grad`` field of each model parameter:

>>> optimizer.zero_grad()

Perform the Jacobian descent backward pass:

>>> torchjd.backward(losses, model.parameters(), A)

This will populate the ``.grad`` field of each model parameter with the corresponding aggregated
Jacobian matrix.

Update each parameter based on its ``.grad`` field, using the ``optimizer``:

>>> optimizer.step()

The model's parameters have been updated!
