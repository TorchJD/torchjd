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
    torchjd.backward([loss1, loss2], aggregator)
    optimizer.step()


def test_iwrm():
    def test_erm_with_sgd():
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

    def test_iwrm_with_ssjd():
        import torch
        from torch.nn import Linear, MSELoss, ReLU, Sequential
        from torch.optim import SGD

        from torchjd import backward
        from torchjd.aggregation import UPGrad

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

    test_erm_with_sgd()
    test_iwrm_with_ssjd()


def test_mtl():
    import torch
    from torch.nn import Linear, MSELoss, ReLU, Sequential
    from torch.optim import SGD

    from torchjd import mtl_backward
    from torchjd.aggregation import UPGrad

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
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)

    import torch
    from lightning import LightningModule, Trainer
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch.nn import Linear, ReLU, Sequential
    from torch.nn.functional import mse_loss
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    from torchjd import mtl_backward
    from torchjd.aggregation import UPGrad

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
