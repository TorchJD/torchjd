"""
This file contains the tests corresponding to the extra usage examples contained in the `.rst` files
of the documentation. When there are multiple examples within a single `.rst` file, we use nested
functions here to test them.
"""


def test_basic_usage():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    import torchjd
    from torchjd.aggregation import UPGrad

    model = Sequential(Linear(10, 5), ReLU(), Linear(5, 2))
    optimizer = SGD(model.parameters(), lr=0.1)

    aggregator = UPGrad()
    input = torch.randn(16, 10)  # Batch of 16 random input vectors of length 10
    target1 = torch.randn(16)  # First batch of 16 targets
    target2 = torch.randn(16)  # Second batch of 16 targets

    loss_fn = MSELoss()
    output = model(input)
    loss1 = loss_fn(output[:, 0], target1)
    loss2 = loss_fn(output[:, 1], target2)

    optimizer.zero_grad()
    torchjd.autojac.backward([loss1, loss2], aggregator)
    optimizer.step()


def test_iwrm():
    def test_autograd():
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

    def test_autogram():
        # TODO: When done fixing this example, we have to copy it in IWRM and adapt the line skips
        #   and highlights
        import torch
        from torch.nn import Linear, MSELoss, ReLU, Sequential
        from torch.optim import SGD

        from torchjd.aggregation import UPGradWeighting
        from torchjd.autogram import GramianReverseAccumulator

        X = torch.randn(8, 16, 10)
        Y = torch.randn(8, 16, 1)

        model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        loss_fn = MSELoss(reduction="none")

        params = model.parameters()
        optimizer = SGD(params, lr=0.1)
        weighting = UPGradWeighting()
        gramian_reverse_accumulator = GramianReverseAccumulator(model)

        for x, y in zip(X, Y):
            y_hat = model(x)
            losses = loss_fn(y_hat, y).squeeze()
            optimizer.zero_grad()
            gramian = gramian_reverse_accumulator.compute_gramian(losses)
            losses.backward(weighting(gramian))
            optimizer.step()

    def test_autojac():
        import torch
        from torch.nn import Linear, MSELoss, ReLU, Sequential
        from torch.optim import SGD

        from torchjd.aggregation import UPGrad
        from torchjd.autojac import backward

        X = torch.randn(8, 16, 10)
        Y = torch.randn(8, 16, 1)

        model = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
        loss_fn = MSELoss(reduction="none")

        params = model.parameters()
        optimizer = SGD(params, lr=0.1)
        aggregator = UPGrad()

        for x, y in zip(X, Y):
            y_hat = model(x)
            losses = loss_fn(y_hat, y)
            optimizer.zero_grad()
            backward(losses, aggregator)
            optimizer.step()

    test_autograd()
    test_autogram()
    test_autojac()


def test_mtl():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import mtl_backward

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


def test_lightning_integration():
    # Extra ----------------------------------------------------------------------------------------
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    # ----------------------------------------------------------------------------------------------

    import torch
    from lightning import LightningModule, Trainer
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch.nn import Linear, ReLU, Sequential
    from torch.nn.functional import mse_loss
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import mtl_backward

    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.feature_extractor = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
            self.task1_head = Linear(3, 1)
            self.task2_head = Linear(3, 1)
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx) -> None:
            input, target1, target2 = batch

            features = self.feature_extractor(input)
            output1 = self.task1_head(features)
            output2 = self.task2_head(features)

            loss1 = mse_loss(output1, target1)
            loss2 = mse_loss(output2, target2)

            opt = self.optimizers()
            opt.zero_grad()
            mtl_backward(losses=[loss1, loss2], features=features, aggregator=UPGrad())
            opt.step()

        def configure_optimizers(self) -> OptimizerLRScheduler:
            optimizer = Adam(self.parameters(), lr=1e-3)
            return optimizer

    model = Model()

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

    dataset = TensorDataset(inputs, task1_targets, task2_targets)
    train_loader = DataLoader(dataset)
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(model=model, train_dataloaders=train_loader)


def test_rnn():
    import torch
    from torch.nn import RNN
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import backward

    rnn = RNN(input_size=10, hidden_size=20, num_layers=2)
    optimizer = SGD(rnn.parameters(), lr=0.1)
    aggregator = UPGrad()

    inputs = torch.randn(8, 5, 3, 10)  # 8 batches of 3 sequences of length 5 and of dim 10.
    targets = torch.randn(8, 5, 3, 20)  # 8 batches of 3 sequences of length 5 and of dim 20.

    for input, target in zip(inputs, targets):
        output, _ = rnn(input)  # output is of shape [5, 3, 20].
        losses = ((output - target) ** 2).mean(dim=[1, 2])  # 1 loss per sequence element.

        optimizer.zero_grad()
        backward(losses, aggregator, parallel_chunk_size=1)
        optimizer.step()


def test_monitoring():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.nn.functional import cosine_similarity
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import mtl_backward

    def print_weights(_, __, weights: torch.Tensor) -> None:
        """Prints the extracted weights."""
        print(f"Weights: {weights}")

    def print_gd_similarity(_, inputs: tuple[torch.Tensor, ...], aggregation: torch.Tensor) -> None:
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
    aggregator.register_forward_hook(print_gd_similarity)

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


def test_amp():
    import torch
    from torch.amp import GradScaler
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGrad
    from torchjd.autojac import mtl_backward

    shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU())
    task1_module = Linear(3, 1)
    task2_module = Linear(3, 1)
    params = [
        *shared_module.parameters(),
        *task1_module.parameters(),
        *task2_module.parameters(),
    ]
    scaler = GradScaler(device="cpu")
    loss_fn = MSELoss()
    optimizer = SGD(params, lr=0.1)
    aggregator = UPGrad()

    inputs = torch.randn(8, 16, 10)  # 8 batches of 16 random input vectors of length 10
    task1_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the first task
    task2_targets = torch.randn(8, 16, 1)  # 8 batches of 16 targets for the second task

    for input, target1, target2 in zip(inputs, task1_targets, task2_targets):
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            features = shared_module(input)
            output1 = task1_module(features)
            output2 = task2_module(features)
            loss1 = loss_fn(output1, target1)
            loss2 = loss_fn(output2, target2)

        scaled_losses = scaler.scale([loss1, loss2])
        optimizer.zero_grad()
        mtl_backward(losses=scaled_losses, features=features, aggregator=aggregator)
        scaler.step(optimizer)
        scaler.update()


def test_partial_jd():
    # TODO: When this is fixed, we need to adapt example accordingly.
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd.aggregation import UPGradWeighting
    from torchjd.autogram import GramianReverseAccumulator

    X = torch.randn(8, 16, 10)
    Y = torch.randn(8, 16, 1)

    model = Sequential(Linear(10, 8), ReLU(), Linear(8, 5), ReLU(), Linear(5, 1))
    loss_fn = MSELoss(reduction="none")

    weighting = UPGradWeighting()

    # Only augment the last part of the model. The weights will be deduced based on the Jacobian
    # only with respect to the parameters contained in this part of the model.
    gramian_reverse_accumulator = GramianReverseAccumulator(model[2:])

    params = model.parameters()
    optimizer = SGD(params, lr=0.1)

    for x, y in zip(X, Y):
        y_hat = model(x)
        losses = loss_fn(y_hat, y).squeeze()
        optimizer.zero_grad()
        gramian = gramian_reverse_accumulator.compute_gramian(losses)
        losses.backward(weighting(gramian))
        optimizer.step()
