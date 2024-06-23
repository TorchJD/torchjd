Stochastic Sub-Jacobian Descent (SSJD)
======================================

This example shows how to use torchjd to perform an iteration of SSJD on a regression model. In this
example, a batch of inputs is forwarded through the model and a matching batch of labels is used to
compute a batch of losses. These losses are then backwarded through the model. The obtained Jacobian
matrix, consisting of the gradients of the losses, is then aggregated using :doc:`UPGrad
<../docs/aggregation/upgrad>` aggregator, and the parameters are updated using the resulting
aggregation.

For the sake of comparison, we provide a single step of SGD on the same example where the loss is
the average over the batch of the individual losses. Note that this would be equivalent to using the
:doc:`Mean <../docs/aggregation/mean>` aggregator with SSJD.


.. grid:: 2

    .. grid-item-card::  SGD

        One iteration of SGD
        ^^^
        .. code-block:: python

            import torch
            from torch.nn import (
                MSELoss,
                Sequential,
                Linear,
                ReLU
            )
            from torch.optim import SGD






            model = Sequential(
                Linear(10, 5),
                ReLU(),
                Linear(5, 1)
            )
            loss_fn = MSELoss()
            params = model.parameters()
            optimizer = SGD(params, lr=0.1)

            x = torch.randn(16, 10)
            y = torch.randn(16, 1)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    .. grid-item-card::  SSJD


        One iteration of SSJD with UPGrad
        ^^^
        .. code-block:: python
            :emphasize-lines: 10, 11, 12, 13, 20, 30

            import torch
            from torch.nn import (
                MSELoss,
                Sequential,
                Linear,
                ReLU
            )
            from torch.optim import SGD

            import torchjd
            from torchjd.aggregation import UPGrad

            A = UPGrad()

            model = Sequential(
                Linear(10, 5),
                ReLU(),
                Linear(5, 1)
            )
            loss_fn = MSELoss(reduction='none')
            params = model.parameters()
            optimizer = SGD(params, lr=0.1)

            x = torch.randn(16, 10)
            y = torch.randn(16, 1)
            y_hat = model(x)
            losses = loss_fn(y_hat, y)

            optimizer.zero_grad()
            torchjd.backward(losses, params, A)
            optimizer.step()
