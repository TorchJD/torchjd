Instance-Wise Multi-Task Learning (IWMTL)
=========================================

TODO

.. code-block:: python

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import Flattening, UPGradWeighting
    from torchjd.autogram import Engine

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]

    loss_fn = MSELoss(reduction="none")
    optimizer = SGD(params, lr=0.1)
    weighting = Flattening(UPGradWeighting())
    engine = Engine(shared_module.modules(), batched_dim=1)

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        features = shared_module(input)
        output1 = task1_module(features).squeeze(1)
        output2 = task2_module(features).squeeze(1)
        losses = torch.stack([loss_fn(output1, target1), loss_fn(output2, target2)])
        gramian = engine.compute_gramian(losses)
        weights = weighting(gramian)

        optimizer.zero_grad()
        losses.backward(weights)
        optimizer.step()
