Instance-Wise Risk Minimization (IWRM)
======================================

This example shows how to use TorchJD to minimize the vector of per-instance losses. This learning
paradigm, called IWRM, is multi-objective, as opposed to the usual empirical risk minimization
(ERM), which seeks to minimize the average loss.

.. hint::
    A proper definition of IWRM and its empirical results on some deep learning tasks are
    available in `Jacobian Descent For Multi-Objective Optimization
    <https://arxiv.org/pdf/2406.16232>`_.

For the sake of the example, we generate a fake dataset consisting of 8 batches of 16 random input
vectors of dimension 10, and their corresponding scalar labels. We train a very simple regression
model to retrieve the label from the corresponding input. To minimize the average loss, we use
stochastic gradient descent (SGD), where each gradient is computed from the average loss over a
batch of data. When minimizing per-instance losses, we use stochastic sub-Jacobian descent, where
each Jacobian matrix consists of one gradient per loss. In this example, we use :doc:`UPGrad
<../docs/aggregation/upgrad>` to aggregate these matrices.

.. grid:: 2

    .. grid-item-card::

        ERM with SGD
        ^^^^^^^^^^^^
        .. code-block:: python
            :emphasize-lines: 21, 29, 31

            import torch
            from torch.nn import (
                Linear,
                MSELoss,
                ReLU,
                Sequential
            )
            from torch.optim import SGD




            X = torch.randn(8, 16, 10)
            Y = torch.randn(8, 16, 1)

            model = Sequential(
                Linear(10, 5),
                ReLU(),
                Linear(5, 1)
            )
            loss_fn = MSELoss()

            params = model.parameters()
            optimizer = SGD(params, lr=0.1)


            for x, y in zip(X, Y):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    .. grid-item-card::

        IWRM with SSJD
        ^^^^^^^^^^^^^^
        .. code-block:: python
            :emphasize-lines: 10-11, 21, 25, 29, 31

            import torch
            from torch.nn import (
                Linear,
                MSELoss,
                ReLU,
                Sequential
            )
            from torch.optim import SGD

            from torchjd import backward
            from torchjd.aggregation import UPGrad

            X = torch.randn(8, 16, 10)
            Y = torch.randn(8, 16, 1)

            model = Sequential(
                Linear(10, 5),
                ReLU(),
                Linear(5, 1)
            )
            loss_fn = MSELoss(reduction='none')

            params = model.parameters()
            optimizer = SGD(params, lr=0.1)
            aggregator = UPGrad()

            for x, y in zip(X, Y):
                y_hat = model(x)
                losses = loss_fn(y_hat, y)
                optimizer.zero_grad()
                backward(losses, aggregator)
                optimizer.step()

Note that in both cases, we use the `torch.optim.SGD
<https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_ optimizer to update the
parameters of the model in the opposite direction of their ``.grad`` field. The difference comes
from how this field is computed.
