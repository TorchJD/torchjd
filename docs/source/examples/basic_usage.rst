Basic Usage
===========

This example shows how to use TorchJD to perform an iteration of Jacobian descent on a regression
model with two objectives. In this example, a batch of inputs is forwarded through the model and two
corresponding batches of labels are used to compute two losses. These losses are then backwarded
through the model. The obtained Jacobian matrix, consisting of the gradients of the two losses with
respect to the parameters, is then aggregated using :doc:`UPGrad <../docs/aggregation/upgrad>`, and
the parameters are updated using the resulting aggregation.



Import several classes from ``torch`` and ``torchjd``:

>>> import torch
>>> from torch.nn import MSELoss, Sequential, Linear, ReLU
>>> from torch.optim import SGD
>>>
>>> import torchjd
>>> from torchjd.aggregation import UPGrad

Define the model and the optimizer, as usual:

>>> model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
>>> optimizer = SGD(model.parameters(), lr=0.1)

Define the aggregator that will be used to combine the Jacobian matrix:

>>> A = UPGrad()

In essence, :doc:`UPGrad <../docs/aggregation/upgrad>` projects each gradient onto the dual cone of
the rows of the Jacobian and averages the results. This ensures that locally, no loss will be
negatively affected by the update.

Now that everything is defined, we can train the model. Define the input and the associated target:

>>> input = torch.randn(16, 10)  # Batch of 16 input random vectors of length 10
>>> target1 = torch.randn(16)  # First batch of 16 targets
>>> target2 = torch.randn(16)  # Second batch of 16 targets

Here, we generate fake inputs and labels for the sake of the example.

We can now compute the losses associated to each element of the batch.

>>> loss_fn = MSELoss()
>>> output = model(input)
>>> loss1 = loss_fn(output[:, 0], target1)
>>> loss2 = loss_fn(output[:, 1], target2)

The last steps are similar to gradient descent-based optimization, but using the two losses.

Reset the ``.grad`` field of each model parameter:

>>> optimizer.zero_grad()

Perform the Jacobian descent backward pass:

>>> torchjd.backward([loss1, loss2], model.parameters(), A)

This will populate the ``.grad`` field of each model parameter with the corresponding aggregated
Jacobian matrix.

Update each parameter based on its ``.grad`` field, using the ``optimizer``:

>>> optimizer.step()

The model's parameters have been updated!
