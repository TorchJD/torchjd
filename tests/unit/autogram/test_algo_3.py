import torch
from torch import Tensor, nn, vmap
from torch.nn import ReLU


def test_algo_3():
    layers = [
        nn.Conv2d(3, 32, 3, bias=False),
        ReLU(),
        nn.Conv2d(32, 64, 3, groups=32, bias=False),
        nn.MaxPool2d(2),
        ReLU(),
        nn.Conv2d(64, 64, 3, groups=64, bias=False),
        nn.MaxPool2d(3),
        ReLU(),
        nn.Flatten(),
        nn.Linear(1024, 128, bias=False),
        ReLU(),
        nn.Linear(128, 1, bias=False),
        nn.Flatten(start_dim=0),
    ]
    batch_size = 16
    input_shape = (batch_size, 3, 32, 32)
    activation = torch.randn(input_shape)

    activations = [activation]
    for layer in layers:
        activation = layer(activation)
        activations.append(activation)

    grad = torch.ones_like(activations[-1])
    gramian = torch.zeros(batch_size, batch_size)
    for i, (input, output, layer) in list(
        enumerate(zip(activations[:-1], activations[1:], layers))
    )[::-1]:
        print("i: ", i)
        params = list(layer.parameters())
        if len(params) > 0:

            def get_vjp(x):
                return torch.autograd.grad(output, params, x, retain_graph=True)

            diagonalized_grad = diagonalize(grad)
            jacobians = vmap(get_vjp)(diagonalized_grad)

            for jacobian in jacobians:
                jacobian = jacobian.reshape((batch_size, -1))
                G = jacobian @ jacobian.T
                print("G: ", G)
                gramian += G

        if i == 0:
            break
        grad = torch.autograd.grad(output, input, grad, retain_graph=True)[0]
        # print("gramian: ", gramian)


def test_diagonalize():
    t = torch.randn([2, 3])
    print(t)
    d = diagonalize(t)
    print(d.shape)
    print(d)


def diagonalize(t: Tensor) -> Tensor:
    d = torch.zeros((t.shape[0],) + t.shape)
    for i, row in enumerate(t):
        d[i, i] = row

    return d
