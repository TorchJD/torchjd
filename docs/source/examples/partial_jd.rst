Partial Jacobian Descent for IWRM
=================================

This example demonstrates how to perform Partial Jacobian Descent using TorchJD. This technique minimizes a vector of
per-instance losses by considering only a submatrix of the Jacobian—specifically, the portion corresponding to a
selected subset of the model's parameters. This approach offers a trade-off between the precision of the aggregation
decision and the computational cost associated with full Jacobian Descent. For a complete, non-partial version, see the
:doc:`IWRM <iwrm>` example.

In this example, our model consists of three `Linear` layers separated by `ReLU` layers. We will perform the partial
descent by considering only the parameters of the last two `Linear` layers and the intervening `ReLU`. By doing this, we
avoid computing the Jacobian and its Gramian with respect to the parameters of the first `Linear` layer, thereby
reducing memory usage and computation time.

.. code-block:: python

    import torch
    from torch.nn import (
        Linear,
        MSELoss,
        ReLU,
        Sequential
    )
    from torch.optim import SGD

    from torchjd import augment_model_for_gramian_based_iwrm
    from torchjd.aggregation import UPGradWeighting

    X = torch.randn(8, 16, 10)
    Y = torch.randn(8, 16, 1)

    model = Sequential(
        Linear(10, 8),
        ReLU(),
        Linear(8, 5),
        ReLU(),
        Linear(5, 1)
    )
    loss_fn = MSELoss()

    weighting = UPGradWeighting()
    augment_model_for_gramian_based_iwrm(model[2:], weighting)

    params = model.parameters()
    optimizer = SGD(params, lr=0.1)

    for x, y in zip(X, Y):
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
