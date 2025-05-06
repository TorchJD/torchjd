Monitoring TorchJD
==================


.. code-block:: python
    :emphasize-lines: 9-11, 13-18, 33-34

    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD
    from torch.nn.functional import cosine_similarity

    from torchjd import mtl_backward
    from torchjd.aggregation import UPGrad

    def print_weights(_, __, weights: torch.Tensor) -> None:
        """Prints the extracted weights."""
        print(f"Weights: {weights}")

    def print_similarity_with_gd(_, inputs: torch.Tensor, aggregation: torch.Tensor) -> None:
        """Prints the cosine similarity between the aggregation and the average gradient."""
        matrix = inputs[0]
        gd_output = matrix.mean(dim=0)
        similarity = cosine_similarity(aggregation, gd_output, dim=0)
        print(f"Cosine similarity: {similarity.item():.4f}")

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]

    loss_fn = MSELoss()
    optimizer = SGD(params, lr=0.1)
    aggregator = UPGrad()

    aggregator.weighting.register_forward_hook(print_weights)
    aggregator.register_forward_hook(print_similarity_with_gd)

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        features = shared_module(input)
        output1 = task1_module(features)
        output2 = task2_module(features)
        loss1 = loss_fn(output1, target1)
        loss2 = loss_fn(output2, target2)

        optimizer.zero_grad()
        mtl_backward(losses=[loss1, loss2], features=features, aggregator=aggregator)
        optimizer.step()
