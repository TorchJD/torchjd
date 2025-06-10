import time
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import Parameter, ReLU

torch.set_default_device("cuda")


def test_algo_3():
    torch.use_deterministic_algorithms(False)

    layers = [
        nn.Conv2d(3, 32, 3),
        ReLU(),
        nn.Conv2d(32, 64, 3, groups=32),
        nn.MaxPool2d(2),
        ReLU(),
        nn.Conv2d(64, 64, 3, groups=64),
        nn.MaxPool2d(3),
        ReLU(),
        nn.Flatten(),
        nn.Linear(1024, 128),
        ReLU(),
        nn.Linear(128, 1),
        nn.Flatten(start_dim=0),
    ]
    batch_size = 4096
    input_shape = (batch_size, 3, 32, 32)
    activation = torch.randn(input_shape)
    target = torch.arange(0, batch_size, dtype=torch.float32)
    criterion = torch.nn.MSELoss(reduction="none")

    activations = [activation]
    for layer in layers:
        activation = layer(activation)
        activations.append(activation)

    losses = criterion(activation, target)
    activations.append(losses)

    start = time.perf_counter()

    # torch.autograd.grad(losses, activations[1:], torch.ones_like(losses))

    grad = torch.ones_like(activations[-1])
    gramian = torch.zeros(batch_size, batch_size)
    for i, (input, output, layer) in list(
        enumerate(zip(activations[:-1], activations[1:], layers + [criterion]))
    )[::-1]:
        params = list(layer.parameters())
        if len(params) > 0:

            # def get_vjp(x):
            #     return torch.autograd.grad(output, params, x, retain_graph=True)

            # diagonalized_grad = diagonalize(grad)
            # jacobian_rows = []
            # for j in range(len(grad)):
            #     # g = diagonalize_one_row(grad, j)
            #     grad_tuple = torch.autograd.grad(output[j], params, grad[j], retain_graph=True)
            #     jacobian_rows.append(torch.concatenate([g.flatten() for g in grad_tuple]))

            def get_vjp(input_j, grad_output_j) -> tuple[Tensor, ...]:
                return vjp_from_module(layer, input_j)(grad_output_j)
                # return torch.autograd.grad(output_j, params, grad_output_j, retain_graph=True)

            jacobians = torch.vmap(get_vjp)(input, grad)

            assert len(jacobians) == 1

            for jacobian in jacobians[0].values():
                J = jacobian.reshape((batch_size, -1))
                gramian += J @ J.T  # Accumulate the gramian

            # jacobian = torch.stack(jacobian_rows)
            # gramian += jacobian @ jacobian.T

            # jacobians = vmap(get_vjp, chunk_size=1)(diagonalized_grad)

            # for jacobian in jacobians:
            #     jacobian = jacobian.reshape((batch_size, -1))
            #     gramian += jacobian @ jacobian.T  # Accumulate the gramian

        if i == 0:
            break  # Don't try to differentiate with respect to the model's input
        grad = torch.autograd.grad(output, input, grad, retain_graph=True)[0]

    end = time.perf_counter()
    print(end - start)
    #
    # start = time.perf_counter()
    #
    # params = OrderedSet(nn.Sequential(*layers).parameters())
    # output = OrderedSet([activations[-1]])
    #
    # init = Init(output)
    # diag = Diagonalize(output)
    # jac = Jac(output, params, chunk_size=1)
    # matrixify = _Matrixify()
    # transform = matrixify << jac << diag << init
    # jacobian_matrices = transform({})
    # exptected_gramian = torch.stack([J @ J.T for J in jacobian_matrices.values()]).sum(dim=0)
    #
    # end = time.perf_counter()
    # print(end - start)
    #
    # assert_close(gramian, exptected_gramian)


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


def diagonalize_one_row(t: Tensor, i: int) -> Tensor:
    d = torch.zeros_like(t)
    d[i] = t[i]

    return d


def test_diagonalize_one_row_sparse():
    t = torch.randn(10, 5)
    print(diagonalize_one_row_sparse(t, 0))
    print(diagonalize_one_row_sparse(t, 1))
    print(diagonalize_one_row_sparse(t, 2))
    print(diagonalize_one_row_sparse(t, 3))


def diagonalize_one_row_sparse(x: Tensor, index: int = 0) -> Tensor:
    """
    Returns a sparse tensor with only the slice at `index` along dim=0 kept, in COO format.
    Works for x.ndim >= 1 of any shape.
    """
    dense = torch.zeros_like(x)
    dense[index] = x[index]
    sparse = dense.to_sparse()  # default is sparse_coo
    return sparse


def vjp_from_module(module: nn.Module, *inputs) -> Callable:
    def functional_model_call(primals: dict[str, Parameter]) -> Tensor:
        all_state = {**primals, **dict(module.named_buffers())}
        return torch.func.functional_call(module, all_state, *inputs)

    return torch.func.vjp(functional_model_call, dict(module.named_parameters()))[1]
