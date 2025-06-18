import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Here's a simple CNN and loss function:


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)


def test_per_sample_grads():
    device = "cuda"
    batch_size = 64
    data = torch.randn(batch_size, 1, 28, 28, device=device)

    targets = torch.randint(10, (64,), device=device)

    model = SimpleCNN().to(device=device)

    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch,))
        loss = loss_fn(predictions, targets)
        return loss

    ft_compute_grad = grad(compute_loss)

    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)


def test_per_sample_grads():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    batch_size = 64

    inputs = torch.randn((batch_size, 5))
    targets = torch.randn((batch_size, 1))

    model = nn.Linear(5, 1)
    params = list(model.parameters())

    outputs = model(inputs)  # shape: [batch_size, 1]
    losses = F.mse_loss(outputs, targets, reduction="none").squeeze()  # shape: [batch_size]

    def compute_one_gradient(loss: Tensor) -> tuple[Tensor, ...]:
        return torch.autograd.grad(loss, params)

    grads = torch.vmap(compute_one_gradient)(losses)
