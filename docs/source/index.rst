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

Packages
========

- **utils**: Contains utility functions such as `backward`.
- **aggregation**: Contains the implementation of aggregators such as UPGrad.

.. toctree::
    :maxdepth: 2
    :name: api reference
    :caption: API Reference
    :hidden:

    packages/utils/index.rst
    packages/aggregation/index.rst

Usage
=====

This example shows how to use torchjd to perform an iteration of Jacobian Descent on a regression
model. More precisely, this is a step of stochastic sub-Jacobian descent where a batch of inputs is
forwarded through the model and the corresponding batch of label is used to build a batch of losses.
These losses are then backwarded through the model and aggregated using UPGrad.

Import several classes from torch and torchjd:

>>> import torch
>>> from torch.nn import MSELoss, Sequential, Linear, ReLU
>>> from torch.optim import SGD
>>>
>>> import torchjd
>>> from torchjd.aggregation import WeightedAggregator, UPGradWrapper, MeanWeighting
>>>
>>> _ = torch.manual_seed(0)  # Set the seed to make this example deterministic

Define the model and the optimizer, as in usual deep learning optimization:

>>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
>>> optimizer = SGD(model.parameters(), lr=0.1)

Define the aggregator that makes a combination of the rows of the jacobian matrices:

>>> W = UPGradWrapper(MeanWeighting())
>>> A = WeightedAggregator(W)

The weights used to make this combination are given by the application of the UPGrad algorithm to
the Jacobian matrix. In short, this algorithm ensures that the parameter update will not
negatively impact any of the losses.

Now that everything is defined, we can train the model. Define the model input and the associated
target:

>>> input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
>>> target = input.sum(dim=1, keepdim=True)  # Batch of 16 targets

Prepare a vector loss for comparing the output of the model to the labels. Setting
`reduction='none'` makes the `MSELoss` into an element-wise loss.

>>> loss = MSELoss(reduction='none')

Here, we generate the data such that each target is equal to the sum of its corresponding input
vector, for the sake of the example.

We can now compute the losses associated to each element of the batch.

>>> output = model(input)
>>> losses = loss(output, target)

The last steps are identical to gradient descent-based optimization.

Reset the ``.grad`` field of each model parameter:

>>> optimizer.zero_grad()

Perform the Jacobian descent backward pass:

>>> torchjd.backward(losses, model.parameters(), A)

This will populate the ``.grad`` field of each model parameter with the corresponding aggregated
Jacobian matrix.

Update each parameter based on its ``.grad`` field, using the ``optimizer``:

>>> optimizer.step()

The model's parameters have been updated!
