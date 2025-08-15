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

TorchJD offers two methods for performing IWRM. The recommended approach, known as the autogram
engine, works by backpropagating the Gramian of the Jacobian of each sample's loss with respect to
the model's parameters. This method is more memory-efficient and generally faster because it avoids
storing the full Jacobians. The resulting Gramian is then used with a specified weighting method for
gradient descent. For more details on this approach, refer to the
:doc:`augment_model <../docs/autogram/augment_model_for_iwrm>` documentation.

The alternative method, the autojac engine, backpropagates the Jacobian of each sample's loss. This
process can be less efficient because the intermediate Jacobians at activation layers are typically
block diagonal and could be compressed, but the autojac engine doesn't perform this optimization. As
a result, it consumes more memory and can lead to a higher computational cost per iteration.

As shown in the code below, any standard PyTorch code can be easily adapted to either of these
approaches.

.. tab-set::
    .. tab-item:: autograd

        Empirical Risk Minimization (ERM) with SGD.

        .. code-block:: python

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD




            X = torch.randn(8, 16, 10)
            Y = torch.randn(8, 16, 1)

            model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
            loss_fn = MSELoss()

            params = model.parameters()
            optimizer = SGD(params, lr=0.1)



            for x, y in zip(X, Y):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    .. tab-item:: autojac

        Instance-Wise Risk Minimization (IWRM) with standard Stochastic Sub-Jacobian Descent.

        .. code-block:: python
            :emphasize-lines: 5-6, 12, 16, 21, 23

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.autojac import backward
            from torchjd.aggregation import UPGrad

            X = torch.randn(8, 16, 10)
            Y = torch.randn(8, 16, 1)

            model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
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

    .. tab-item:: autogram (recommended)

        Instance-Wise Risk Minimization (IWRM) with Gramian-based Stochastic Sub-Jacobian Descent.

        .. code-block:: python
            :emphasize-lines: 5-6, 16-17

            import torch
            from torch.nn import Linear, MSELoss, ReLU, Sequential
            from torch.optim import SGD

            from torchjd.autogram import augment_model_for_iwrm
            from torchjd.aggregation import UPGradWeighting

            X = torch.randn(8, 16, 10)
            Y = torch.randn(8, 16, 1)

            model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
            loss_fn = MSELoss()

            params = model.parameters()
            optimizer = SGD(params, lr=0.1)
            weighting = UPGradWeighting()
            augment_model_for_iwrm(model, weighting)

            for x, y in zip(X, Y):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        The per-sample gradients are never fully stored in memory, leading to large improvements in
        memory usage and speed in most practical cases. The results should be the same as with
        autojac (up to tiny numerical imprecisions), as long as the model always treats each
        instance independently from other instances in the batch (e.g. no batch-normalization is
        used).

Note that in all three cases, we use the `torch.optim.SGD
<https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_ optimizer to update the
parameters of the model in the opposite direction of their ``.grad`` field. The difference comes
from how this field is computed.
